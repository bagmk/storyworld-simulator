"""
Prose Generator for the AI Story Simulation Engine.

Generates literary-quality prose from distilled scenes + original YAML beats.
This replaces the old novel_generator.py pipeline that worked from raw turn logs.

Key differences from novel_generator.py:
  - Reads original YAML beats directly (not just simulation output)
  - Works from DistilledScene objects (compressed, deduplicated)
  - Generates prose per-scene with beat-aware context
  - Single coherent LLM call per scene (not per-chunk of turns)
  - Controls tone, pacing, and episode position explicitly

Pipeline:
  1. Receive list of DistilledScene objects + episode config
  2. For each scene: generate literary prose grounded in YAML beats + distilled content
  3. Generate internal monologue transitions between scenes
  4. Combine into a single chapter
  5. Polish for consistency and target word count
  6. Write final chapter as Markdown
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from .polisher import ChapterPolisher
from .reader_profile import ReaderProfile, build_reader_profile
from .scene_distiller import DistilledScene
from . import database as db
from .review_feedback import build_feedback_prompt_block, count_feedback_term_occurrences

logger = logging.getLogger(__name__)

# Hard rule: never output screenplay-style speaker labels in chapter prose.
COLON_DIALOGUE_LABEL_BAN = (
    "- 본문에서 `이름: \"대사\"` 형식(예: `수민: \"...\"`)을 절대 사용하지 말 것.\n"
)


class ProseGenerator:
    """
    Generates literary prose from distilled scenes and YAML beat definitions.

    Parameters
    ----------
    llm : LLMClient
        Used with premium model for narrative generation.
    episode_config : dict
        Original YAML episode configuration.
    output_dir : str
        Directory where chapter .md files are saved.
    """

    def __init__(
        self,
        llm: LLMClient,
        episode_config: dict,
        output_dir: str = "output",
        character_profiles: Optional[list[dict]] = None,
        previous_episode_context: Optional[str] = None,
        include_all_episode_context: bool = True,
        max_history_episodes: Optional[int] = None,
        runtime_policy: Optional[dict] = None,
        reader_feedback: Optional[dict] = None,
        guardian_briefing: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.episode_config = episode_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.character_profiles = character_profiles or []
        self.character_index = self._build_character_index(self.character_profiles)
        self.previous_episode_context = previous_episode_context
        self.include_all_episode_context = include_all_episode_context
        self.runtime_policy = runtime_policy or {}
        self.reader_profile: ReaderProfile = build_reader_profile(reader_feedback)
        self.reader_feedback = self.reader_profile.as_dict()
        self.guardian_briefing = guardian_briefing or ""
        self.chapter_polisher = ChapterPolisher(
            llm=llm,
            episode_config=self.episode_config,
            runtime_policy=self.runtime_policy,
            reader_feedback=self.reader_feedback,
        )
        if max_history_episodes is None and self.runtime_policy.get("prose_history_max_episodes") is not None:
            self.max_history_episodes = int(self.runtime_policy.get("prose_history_max_episodes"))
        else:
            self.max_history_episodes = max_history_episodes

    # ------------------------------------------------------------------ #
    # Public: Generate Chapter
    # ------------------------------------------------------------------ #

    def generate_chapter(
        self,
        scenes: list[DistilledScene],
        protagonist_name: str = "Kim Sumin",
        style: str = "third_person_close",
        target_words: int = 3500,
    ) -> str:
        """
        Generate a full novel chapter from distilled scenes.

        Returns path to the generated .md file.
        """
        episode_id = self.episode_config.get("id", "unknown")
        logger.info(
            "Generating chapter for %s: %d scenes, target %d words",
            episode_id, len(scenes), target_words,
        )

        # Climax scenes get 30% more, opening/resolution 15% less
        scene_budgets = self._calculate_scene_budgets(scenes, target_words)

        # Build episode-level context
        episode_context = self._build_episode_context(protagonist_name)
        continuity_context = self._build_previous_episode_context(episode_id)

        # Generate title
        title = self._generate_title(scenes, episode_context)

        # Generate prose for each scene
        prose_sections: list[str] = []
        established_anchors: set[str] = set()
        for i, scene in enumerate(scenes):
            prev_section = prose_sections[-1] if prose_sections else None
            scene_anchor_source = "\n".join(
                [d for d in scene.discoveries if isinstance(d, str)]
                + [scene.narrative_summary or ""]
            )
            scene_anchors = self._extract_anchor_terms(scene_anchor_source)
            section = self._generate_scene_prose(
                scene=scene,
                scene_index=i,
                total_scenes=len(scenes),
                episode_context=episode_context,
                protagonist_name=protagonist_name,
                style=style,
                word_budget=scene_budgets[i],
                prev_section_tail=prev_section[-300:] if prev_section else None,
                previous_episode_context=continuity_context,
                established_anchors=sorted(established_anchors)[:24],
            )
            prose_sections.append(section)
            established_anchors.update(scene_anchors[:12])
            logger.info(
                "Scene %d/%d '%s': %d words",
                i + 1, len(scenes), scene.title, len(section.split()),
            )

        chapter_anchors = self._collect_episode_anchor_terms(episode_context)
        coverage_anchors = self._select_anchor_terms_for_coverage(chapter_anchors)
        coverage_anchors = self._tune_coverage_anchors(coverage_anchors)
        combined = self._combine_with_transitions(
            prose_sections, scenes, episode_context, style
        )

        final = self.chapter_polisher.polish_chapter(
            combined,
            target_words=target_words,
            style=style,
            protagonist_name=protagonist_name,
            chapter_anchors=coverage_anchors,
            prose_adapter=self,
        )
        diagnostics = self._collect_style_diagnostics(final)
        logger.info(
            "Style diagnostics | avg_par_sent=%.2f long_sent_ratio=%.2f jargon_repeats=%d sensory_streak=%d",
            diagnostics.get("avg_paragraph_sentences", 0.0),
            diagnostics.get("long_sentence_ratio", 0.0),
            diagnostics.get("jargon_repeat_terms", 0),
            diagnostics.get("max_visual_streak", 0),
        )

        # Write
        out_path = self.output_dir / f"{episode_id}_chapter.txt"
        self._write_chapter(out_path, title, final, episode_id, scenes)

        word_count = len(final.split())
        logger.info("Chapter written: %s (%d words)", out_path, word_count)
        return str(out_path)

    # ------------------------------------------------------------------ #
    # Episode Context
    # ------------------------------------------------------------------ #

    def _build_episode_context(self, protagonist_name: str) -> dict:
        """Build rich episode context from YAML config."""
        ep = self.episode_config
        ep_id = str(ep.get("id", ""))

        # Extract episode number
        ep_num = 0
        for part in ep_id.split("_"):
            digits = "".join(c for c in part if c.isdigit())
            if digits:
                ep_num = int(digits)
                break

        pacing = ep.get("pacing", "normal")
        pacing_tone = {
            "slow": "contemplative, observational, rich in sensory detail and internal reflection",
            "normal": "balanced between action and reflection, natural rhythm",
            "tense": "tight, anxious, every detail feels loaded with significance",
            "fast": "urgent, compressed, events cascade without time to process",
        }.get(pacing, "balanced")

        # Beat summaries
        clues = ep.get("introduced_clues", [])
        beats = [
            {
                "id": str(c.get("id", "")).strip(),
                "content": str(c.get("content", "")).strip(),
            }
            for c in clues
            if isinstance(c, dict) and str(c.get("id", "")).strip()
        ]
        beat_by_id = {b["id"]: b["content"] for b in beats if b.get("content")}

        return {
            "episode_number": ep_num,
            "total_episodes": 49,
            "location": ep.get("location", ""),
            "date": ep.get("date", ""),
            "summary": ep.get("summary", ""),
            "pacing": pacing,
            "pacing_tone": pacing_tone,
            "protagonist": protagonist_name,
            "beats": beats,
            "beat_by_id": beat_by_id,
            "recommended_length": ep.get("recommended_length", 3500),
        }

    @staticmethod
    def _norm_char_key(value: str) -> str:
        return re.sub(r"[^0-9a-zA-Z가-힣]+", "", str(value or "").lower())

    def _build_character_index(self, profiles: list[dict]) -> dict[str, dict]:
        index: dict[str, dict] = {}
        for row in profiles:
            if not isinstance(row, dict):
                continue
            speech = row.get("speech_profile", {}) or {}
            visual = row.get("visual_profile", {}) or {}
            if not isinstance(speech, dict):
                speech = {}
            if not isinstance(visual, dict):
                visual = {}
            if not speech and not visual:
                continue
            keys = [
                str(row.get("id", "")).strip(),
                str(row.get("name", "")).strip(),
            ]
            aliases = row.get("aliases", [])
            if isinstance(aliases, list):
                keys.extend(str(a).strip() for a in aliases)
            for key in keys:
                norm = self._norm_char_key(key)
                if norm and norm not in index:
                    index[norm] = {
                        "speech_profile": speech,
                        "visual_profile": visual,
                    }
        return index

    def _character_profile_for_name(self, name: str) -> dict:
        norm = self._norm_char_key(name)
        if not norm:
            return {}
        return self.character_index.get(norm, {})

    def _build_scene_character_guide(self, character_names: list[str]) -> str:
        lines: list[str] = []
        for name in character_names:
            profile = self._character_profile_for_name(name)
            if not isinstance(profile, dict):
                continue
            parts = []
            speech = profile.get("speech_profile", {}) or {}
            visual = profile.get("visual_profile", {}) or {}
            tone = str(speech.get("tone", "")).strip()
            cadence = str(speech.get("cadence", "")).strip()
            formality = str(speech.get("formality", "")).strip()
            lexicon = speech.get("lexicon", []) or []
            tics = speech.get("signature_tics", []) or []
            avoid = speech.get("avoid", []) or []
            wardrobe = str(visual.get("wardrobe", "")).strip()
            silhouette = str(visual.get("silhouette", "")).strip()
            body_language = str(visual.get("body_language", "")).strip()
            vibe = str(visual.get("vibe", "")).strip()
            if tone:
                parts.append(f"tone={tone}")
            if cadence:
                parts.append(f"cadence={cadence}")
            if formality:
                parts.append(f"formality={formality}")
            if lexicon:
                parts.append(f"lexicon[{', '.join(str(x) for x in lexicon[:6])}]")
            if tics:
                parts.append(
                    f"optional_tics(sparingly,max1)[{', '.join(str(x) for x in tics[:2])}]"
                )
            if avoid:
                parts.append(f"avoid[{', '.join(str(x) for x in avoid[:4])}]")
            if wardrobe:
                parts.append(f"wardrobe={wardrobe}")
            if silhouette:
                parts.append(f"silhouette={silhouette}")
            if body_language:
                parts.append(f"body_language={body_language}")
            if vibe:
                parts.append(f"vibe={vibe}")
            if parts:
                lines.append(f"- {name}: " + " | ".join(parts))
        return "\n".join(lines)

    def _build_previous_episode_context(self, episode_id: str) -> str:
        """
        Build cross-episode continuity context.
        Priority:
          1) explicit `previous_episode_context` argument
          2) auto-generated summary from all prior completed episodes
        """
        manual = (self.previous_episode_context or "").strip()
        if manual:
            return manual
        if not self.include_all_episode_context:
            return ""

        try:
            history = db.load_episode_history_context(
                current_episode_id=episode_id,
                max_episodes=self.max_history_episodes,
            )
        except Exception:
            logger.exception("Failed to load episode history context")
            return ""

        if not history:
            return ""

        lines = ["Cross-episode memory (chronological):"]
        for item in history:
            eid = str(item.get("id", "")).strip()
            if not eid:
                continue
            date = str(item.get("date", "")).strip() or "date-unknown"
            location = str(item.get("location", "")).strip() or "location-unknown"
            summary = self._truncate_text(str(item.get("summary", "")).strip(), 140)
            clue_ids = item.get("clue_ids", [])
            clue_preview = ", ".join(clue_ids[:4]) if isinstance(clue_ids, list) and clue_ids else "-"
            lines.append(
                f"- {eid} | {date} | {location} | {summary} | clues: {clue_preview}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Scene Prose Generation
    # ------------------------------------------------------------------ #

    def _generate_scene_prose(
        self,
        scene: DistilledScene,
        scene_index: int,
        total_scenes: int,
        episode_context: dict,
        protagonist_name: str,
        style: str,
        word_budget: int,
        prev_section_tail: Optional[str] = None,
        previous_episode_context: Optional[str] = None,
        established_anchors: Optional[list[str]] = None,
    ) -> str:
        """Generate literary prose for one distilled scene."""
        style = self._effective_style(style)
        pov = "first person" if style == "first_person" else "third person close"
        ep = episode_context
        protagonist_short = "수민" if "sumin" in protagonist_name.lower() else protagonist_name
        date_anchor = str(ep.get("date", "")).strip()

        # Format scene content
        has_source_dialogue = bool(scene.key_dialogue)
        dialogue_text = "\n".join(
            f"  {d.get('speaker', '?')}: \"{d.get('line', '')}\""
            for d in scene.key_dialogue
        ) or "  (no key dialogue in simulation data)"

        actions_text = "\n".join(
            f"  - {a}" for a in scene.key_actions
        ) or "  (no key actions)"

        discoveries_text = "\n".join(
            f"  - {d}" for d in scene.discoveries
        ) or "  (no discoveries)"

        # Beat context — what YAML says should happen
        beat_context = ""
        matched_beats: list[tuple[str, str]] = []
        if scene.beat_references:
            matched_beats = [
                (ref, ep.get("beat_by_id", {}).get(ref, ""))
                for ref in scene.beat_references
                if ep.get("beat_by_id", {}).get(ref, "")
            ]
            if matched_beats:
                beat_context = "Original story beats for this scene:\n" + "\n".join(
                    f"  - [{bid}] {btxt}" for bid, btxt in matched_beats
                )

        # Extract concrete anchors (numbers/codenames/proper terms) that must survive.
        anchor_source = "\n".join(
            [d for d in scene.discoveries if isinstance(d, str)]
            + [btxt for _, btxt in matched_beats if btxt]
        )
        anchors = self._extract_anchor_terms(anchor_source)
        anchors_text = ", ".join(anchors[:16]) if anchors else "(none)"
        established = [a.strip() for a in (established_anchors or []) if isinstance(a, str) and a.strip()]
        established_lower = {a.lower() for a in established}
        recalled = [a for a in anchors if a.lower() in established_lower][:10]
        new_anchors = [a for a in anchors if a.lower() not in established_lower][:10]
        recalled_text = ", ".join(recalled) if recalled else "(none)"
        new_anchor_text = ", ".join(new_anchors) if new_anchors else "(none)"

        # Position description
        if scene_index == 0:
            position = "OPENING — establish atmosphere, setting, and protagonist's state of mind"
        elif scene_index == total_scenes - 1:
            position = "CLOSING — bring threads together, leave resonance, hint at what's next"
        elif scene.pacing == "climax":
            position = "CLIMAX — this is the pivotal moment; give it full emotional weight"
        else:
            position = f"MIDDLE (scene {scene_index + 1}/{total_scenes}) — develop naturally"

        readability = self._readability_controls()
        sentence_cap = self._feedback_sentence_word_cap(default=25)
        sensory_cap = self._feedback_sensory_channel_cap(default=2)
        emotion_repeat_cap = self._feedback_emotion_repeat_cap(default=1)
        term_glossary = self._build_scene_term_glossary(scene, matched_beats, blocked_terms=established)

        system = (
            "You are writing a Korean serialized techno-thriller chapter scene.\n"
            "Prioritize dramatic flow, subtext, and character individuality over metrics.\n"
            "No benchmark-style quotas or checklists. Write as a real novel scene.\n"
            "Avoid repetitive dialogue tags like '말했다/물었다' in consecutive lines; "
            "prefer action beats, gaze shifts, silence, interruption, and sentence rhythm to track speakers.\n"
            "Do not turn stage directions or narration into quoted speech.\n"
            "Use signature verbal tics only when context clearly demands it; avoid catchphrase repetition.\n"
            "Let each character's speed, hesitation, and directness feel distinct when they speak.\n"
            "When multiple speakers press the same issue, make each line do a different job: one opens leverage, one counters, one narrows the choice.\n"
            "Use concrete sensory details only at pressure turns; do not layer similar sensations repeatedly.\n"
            "When a pressure beat already has one bodily reaction, do not add another near-identical reaction in the next sentence; move to choice, consequence, or dialogue instead.\n"
            "If the pressure has already landed, do not keep narrating the same hesitation; cash it out through a concrete action or scene boundary.\n"
            "If a threat stays abstract, anchor it once in a person, object, rule, or visible room reaction instead of repeating danger language.\n"
            "Treat repeated atmosphere as expendable; one sharp image is enough when the scene is already tense.\n"
            "If a later sentence repeats the same observation, bridge, or reaction with a new opener, prefer the version that moves the scene forward.\n"
            "When a decision lands, show it once through a small habitual motion or concrete action before the next line.\n"
            "If an important bystander or authority figure is still unnamed, keep one neutral role cue once and reuse it consistently; do not alternate between a vague label and a proper name without a clear bridge.\n"
            "Keep all content in Korean.\n"
        )
        pov_and_time = (
            f"- POV: {pov}. Narration center is {protagonist_short}.\n"
            "- Keep chronology and location transitions readable in prose.\n"
            f"- Date anchor if relevant: {date_anchor if date_anchor else '(none)'}.\n"
        )
        scene_character_guide = self._build_scene_character_guide(scene.characters_present)

        continuity = ""
        if prev_section_tail:
            continuity = (
                f"\n## Previous Section Ending\n"
                f"...{prev_section_tail}\n"
                f"Continue naturally from this point.\n"
            )

        prompt = (
            f"## Episode Context\n"
            f"Episode {ep['episode_number']}/{ep['total_episodes']}\n"
            f"Location: {ep['location']}\n"
            f"Date anchor: {date_anchor if date_anchor else '(none)'}\n"
            f"Pacing: {ep['pacing']} — {ep['pacing_tone']}\n"
            f"Protagonist: {ep['protagonist']}\n\n"
            f"## Scene: {scene.title}\n"
            f"Position: {position}\n"
            f"Emotional arc: {scene.emotional_arc}\n"
            f"Location: {scene.location}\n"
            f"Characters: {', '.join(scene.characters_present)}\n\n"
            f"## POV and Time Guidance\n{pov_and_time}\n"
            f"## Readability and Rhythm Constraints\n"
            f"- Keep paragraph rhythm breathable: usually {readability['paragraph_min']}-{readability['paragraph_max']} sentences per paragraph.\n"
            f"- Keep most sentences under about {sentence_cap} words; split explanatory chains before they sprawl.\n"
            f"- Alternate inner thought and outer action to avoid continuous analytical voice.\n"
            f"- Reserve short sentence beats for genuine pressure turns; otherwise keep nearby clauses connected.\n"
            f"- Avoid repeating the same tension phrasing across nearby paragraphs.\n\n"
            f"- If a reaction has already been shown, do not restate it with the same emotional wording; switch to a different visible cue such as gaze, hand movement, posture, breath, or room reaction.\n"
            f"- Avoid repeating identical numeric literals (e.g., same milliseconds/ratios) in adjacent paragraphs unless plot-critical.\n\n"
            f"- In dialogue-heavy stretches, anchor speaker identity with short action beats or name cues every 1-2 exchanges.\n"
            f"- When two speakers make similar warnings or proposals, keep their roles distinct by shifting one into leverage, one into resistance, and one into consequence.\n"
            f"- When several supporting characters are present, give each one a distinct immediate motive, pressure, or threat in one concrete line instead of repeating the same support/control idea.\n"
            f"- If three or more characters are present, avoid ambiguous pronouns for consecutive lines.\n\n"
            f"- Technical terms: explain only if the scene becomes unclear, and prefer brief inline cues over parentheses.\n"
            f"- Rotate sentence length in short/medium/long rhythm to avoid monotone cadence.\n"
            f"- If explanation runs long, pivot with one concrete action or reaction sentence instead of another detached fragment.\n"
            f"- If 2-3 short narrative sentences describe one beat, fuse them into one flowing sentence with clear cause/effect.\n"
            f"- Let only one strong interior realization carry a local beat; after that, pivot into action, reaction, or verbal collision.\n"
            f"- If a worry, explanation, or inner question already landed once, do not restate it in the next sentence; move to consequence, movement, or dialogue.\n"
            f"- Do not restate the same situation, emotion, or image in consecutive paragraphs unless new stakes changed it.\n"
            f"- When tension is already established, advance with one new choice, discovery, or reaction instead of repeating the same pressure line.\n"
            f"- Once a fact, threat, or decision lands, the next sentence should show consequence, reaction, or movement instead of paraphrasing it.\n"
            f"- For any pressure beat, keep at most one bodily or emotional reaction sentence before the scene changes state.\n"
            f"- Save the hardest tension wording for only one or two decisive beats in the scene.\n"
            f"- If a threat is present, name one concrete anchor for it once; do not keep restating abstract danger in different words.\n"
            f"- If the scene includes multiple supporting characters, let each one carry a different pressure or motive so they do not blur into one repeated concern.\n"
            f"- If repeated atmosphere starts to stack, keep the sharpest image and cut the rest.\n"
            f"- If a fact, fear, or interpretation already landed once, do not paraphrase it in the next sentence; move to reaction, interruption, or decision.\n"
            f"- If scene focus changes, open the paragraph with the acting subject or a short location cue so the reader does not have to reconstruct who moved first.\n"
            f"- If similar sensory channel repeats for recent 3+ sentences, switch to another channel (sound/touch/temperature).\n"
            f"- Expository dialogue should be compressed; let one factual line land, then follow with a micro-action, reaction, or room cue instead of repeating the explanation.\n\n"
            f"- If two nearby paragraphs perform the same job (restating a question, explaining the same fact, lingering on the same pressure), compress them into one sharper paragraph.\n"
            f"- Keep local event axis linear: question -> response -> approach/offer should appear once in that order, not as repeated resets.\n"
            f"- Avoid stock time bridges like '그 직후' or '잠시 뒤'; prefer gaze shift, footsteps, door movement, microphone lowering, or location cue.\n\n"
            f"- Do not stack multiple warning-style signals in one short span unless one directly triggers the next; keep the sharpest cue and cash it out through reaction.\n"
            f"- Keep sensory description to about {sensory_cap} channels per paragraph, and save the sharpest image for the most important beat.\n"
            f"- Do not reuse the same emotion word or paraphrase more than about {emotion_repeat_cap} time per local beat.\n"
            f"- If an English keyword or technical term appears, use at most one short plain Korean cue when comprehension needs it, then move to reaction, action, or decision instead of repeating the explanation.\n"
            f"- Terms like latency or real-time should get only one short onboarding explanation in the chapter; later mentions should shift quickly into {protagonist_short}의 판단, 감정, 또는 선택.\n"
            f"- If technical wording starts to crowd out the beat, collapse it into one visible action, object cue, or body reaction instead of another explanatory paraphrase.\n"
            f"- Do not explain the same technical idea in two adjacent sentences; one plain-language cue is enough.\n"
            f"- If commas or connectives start chaining clauses, cut the sentence into shorter direct beats.\n\n"
            f"- Keep one dominant axis per sentence: action, perception, or emotion. If two axes matter, split them into explicit cause/effect.\n"
            f"- If you use one metaphor or comparison, do not spend the next sentence unpacking it again; move straight to reaction or the next action.\n"
            f"- When a memo, warning sound, or named arrival shifts the scene, give it one dedicated sentence with subject and location.\n\n"
            f"- Do not lean on default connective openers like '그리고', '그러자', '다만' in nearby lines; vary or omit them when flow allows.\n"
            f"- If two clipped beats belong to one moment, fuse them into one cause/effect sentence instead of reusing a stock bridge opener.\n"
            f"- If a bridge phrase has already carried the beat, replace the next one with a concrete action, gaze, object, or room-state cue.\n"
            f"- Mix forceful pressure lines with plainer connective sentences so the rhythm rises and settles instead of staying equally taut.\n\n"
            f"- Let character agendas and speech habits differentiate dialogue; theme should emerge from friction, not repeated explanation.\n"
            f"- If two speakers sound alike, make one line carry the offer or warning and the other carry the reply or cost instead of repeating the same psychological point.\n"
            f"- Use the character guide's cadence, formality, lexicon, and optional tics as real voice cues, not decoration.\n\n"
            f"{COLON_DIALOGUE_LABEL_BAN}\n"
            f"## Essential Content (from simulation)\n"
            f"Key dialogue:\n{dialogue_text}\n"
            f"Dialogue source status: {'simulation key dialogue exists' if has_source_dialogue else 'simulation key dialogue sparse; infer naturally'}\n\n"
            f"Key actions:\n{actions_text}\n\n"
            f"Discoveries/revelations:\n{discoveries_text}\n\n"
            f"Scene summary: {scene.narrative_summary}\n\n"
        )
        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "이름이 반복", "인물", "역할", "구분", "호칭", "이름", "말투", "어투", "톤", "speaker"):
            prompt += (
                "## Speaker Clarity Priority\n"
                "- In every dialogue cluster, attach explicit speaker/addressee cues.\n"
                "- Avoid back-to-back ambiguous pronoun-only dialogue lines.\n"
                "- Keep naming stable for known characters; avoid unnecessary title/name switching.\n"
                "- Do not reintroduce already-known characters with repetitive identity labels.\n\n"
            )
        if self._feedback_mentions("말투", "어투", "톤", "대화 톤", "고유한 말투"):
            prompt += (
                "## Dialogue Voice Variety Priority\n"
                "- Different characters should not sound identical in sentence ending and lexical rhythm.\n"
                "- Add subtle per-character speech habits without caricature.\n"
                "- Let one speaker be clipped, another more hesitant, and another more direct when the scene pressure changes.\n"
                "- When several speakers address the same issue, vary the function of each line so the same concern is not repeated in different words.\n"
                "- If a line starts sounding like exposition, attach it to a tactic, dodge, interruption, or reaction beat so it still feels spoken.\n"
                "- Avoid repeating the same sentence-ending pattern across adjacent dialogue lines.\n\n"
            )
        if self._feedback_flag_enabled("dialogue_agenda_contrast") or self._feedback_mentions("이해관계", "말버릇"):
            prompt += (
                "## Dialogue Agenda Contrast Priority\n"
                "- Let each major speaker push a concrete immediate agenda rather than restating theme.\n"
                "- Differentiate voices through directness, evasiveness, brevity, and favored wording.\n"
                "- When support, control, or pressure repeats across nearby lines, vary the leverage or consequence instead of paraphrasing the same motive.\n"
                "- Once a motive is clear, make later tension come from collision, concession, or pressure change.\n\n"
            )
        if self.runtime_policy.get("hold_pressure_peak") or self.runtime_policy.get("prefer_concrete_offer_detail") or self.runtime_policy.get("prefer_concrete_threat_detail") or self._feedback_mentions("위험", "대가", "통제", "계약", "경비", "시설", "보안", "배지", "명함", "조항"):
            prompt += (
                "## Pressure Cash-Out Priority\n"
                "- When a coercive offer or pressure tactic appears, do not leave the cost abstract.\n"
                "- Show the price through concrete scene details: clause wording, access restriction, security, badge, facility rule, room placement, deadline, a one-line usage instruction, or a visible handoff.\n"
                "- If the scene already includes one body reaction, do not add another mirrored body beat; let the next sentence carry the consequence or forced choice.\n"
                "- If the source material only implies danger, keep the implication concrete and physical; do not explain the threat in summary language.\n"
                "- After the offer lands, the next beat should be consequence, refusal, or a forced choice, not another paraphrase of the warning.\n\n"
            )
        if self._feedback_mentions("문장 구조", "반복적인 문장 구조", "비슷한 리듬", "같은 리듬", "단조", "지루", "반복되는 표현", "묘사가 반복"):
            prompt += (
                "## Sentence Variety Priority\n"
                "- Vary sentence openings and clause shapes; do not let 3 nearby sentences start with the same subject/action pattern.\n"
                "- Mix short, medium, and longer sentences with deliberate contrast instead of repeating clipped declarative beats.\n"
                "- When using a short sentence for tension, surround it with context-rich sentences so rhythm does not turn mechanical.\n\n"
            )
        if self._feedback_mentions("간결한 문장", "문맥 파악", "맥락 파악", "문맥", "맥락", "따라가기 힘들", "길고 복잡", "이해하기 어려", "이해하기 어렵"):
            prompt += (
                "## Context Link Priority\n"
                "- Keep subject, place, and causal link explicit whenever a sentence becomes short.\n"
                "- Avoid stacking multiple ultra-short sentences that omit who acted or why it matters.\n"
                "- If a clipped sentence would feel ambiguous alone, fuse it with the neighboring sentence.\n\n"
            )
        if self._feedback_reports_stalled_progression():
            prompt += (
                "## Stalled Progression Priority\n"
                "- If a beat already landed, do not spend another paragraph analyzing or restating the same pressure.\n"
                "- Move from tension to decision, interruption, discovery, or location shift within the next 1-2 sentences.\n"
                "- End paragraphs on changed situation or consequence, not repeated atmosphere.\n\n"
            )
        if self._feedback_mentions("전개가 느려", "느려서 집중", "집중력을 잃", "늘어지", "템포가 느려", "속도감이 떨어"):
            prompt += (
                "## Story Momentum Priority\n"
                "- Each paragraph must change something: decision, tension, discovery, or relationship pressure.\n"
                "- Remove restatement of the same concern once it has landed.\n"
                "- If a conversation beat explains too long, compress it and move to reaction or consequence.\n\n"
            )
        if self._feedback_mentions("반복되는 표현", "비슷한 상황", "비슷한 상황과 묘사", "묘사가 반복", "지루"):
            prompt += (
                "## Anti-Repetition Priority\n"
                "- Keep only the freshest local image or phrase for a repeated tension beat.\n"
                "- If a paragraph repeats a situation already established, convert it into one short consequence sentence instead.\n"
                "- Do not reuse the same emotional paraphrase across adjacent paragraphs.\n\n"
            )
        if self._feedback_mentions("비슷한 감각 묘사", "감각 묘사", "심리 표현", "읽는 속도", "속도가 조금 처지"):
            prompt += (
                "## Sensory Compression Priority\n"
                "- Cut repeated sensory and inner-response description aggressively.\n"
                "- Keep one sharp sensory image for the beat that truly changes pressure; compress the rest into action or consequence.\n"
                "- If a technical or English phrase appears, show the character's immediate feeling or body reaction right away.\n\n"
            )
        if self._feedback_mentions("심리", "내면", "설명적", "감정선", "표정", "행동", "보여"):
            prompt += (
                "## Emotion Delivery Priority\n"
                "- Show emotion through micro-actions, gaze, breath, posture, and response timing.\n"
                "- Avoid abstractly explaining feelings for multiple consecutive sentences.\n"
                "- Keep one short inner-thought line only when needed after concrete action.\n\n"
            )
        if self._feedback_mentions("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트"):
            prompt += (
                "## Dialogue Impact Priority\n"
                "- Expository dialogue should be compressed into one short factual line.\n"
                "- Follow factual dialogue with reaction/action/subtext to keep emotional impact.\n"
                "- Avoid chaining multiple explanation clauses inside one quote.\n\n"
            )
        if self._feedback_mentions("긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려"):
            prompt += (
                "## Long Dialogue Pacing Priority\n"
                "- In long discussion stretches, alternate quotes with short action or setting reaction beats.\n"
                "- Avoid extended back-to-back expository quotes without movement.\n"
                "- Keep only one key explanatory quote per local beat and compress the rest.\n\n"
            )
        if self._feedback_mentions("장면 전환", "전환", "복도", "발표장", "흐름"):
            prompt += (
                "## Transition Clarity Priority\n"
                "- When scene location or focus shifts, insert one short transition sentence.\n"
                "- Keep transition lines concrete and action-based, not analytical.\n\n"
            )
        if self.runtime_policy.get("prefer_concrete_transition_cue"):
            prompt += (
                "## Concrete Transition Cue Priority\n"
                "- Make the bridge carry one specific movement, object, or doorway cue so the reader can see the handoff.\n"
                "- Do not use a vague connective sentence when an observable action can do the job.\n\n"
            )
        if self._feedback_flag_enabled("compress_threat_signal_stack"):
            prompt += (
                "## Threat Signal Density Priority\n"
                "- Memo discovery, monitor alert, and watchful-observer cues should not pile up in one paragraph unless one clearly triggers the next.\n"
                "- Keep the sharpest warning cue on the page, then turn the rest into reaction, movement, or confrontation.\n\n"
            )
        if self._feedback_flag_enabled("prefer_linear_scene_axis") or self._feedback_flag_enabled("prioritize_chronological_scene_order") or self._feedback_mentions("시간축", "시간 순서", "순서가 섞", "되감기", "헷갈리", "단선 구조"):
            prompt += (
                "## Linear Scene Axis Priority\n"
                "- Keep event order strictly linear inside the scene.\n"
                "- If a question becomes an answer and then an approach or offer, do not rewind to the earlier question stage.\n"
                "- Once movement leaves the podium, stage, or room edge, make the next paragraph continue from that new position.\n\n"
            )
        if self._feedback_flag_enabled("clarify_event_transitions") or self._feedback_mentions("메모 발견", "경고음", "밀러 등장", "공간 동선", "인물 위치"):
            prompt += (
                "## Event Turn Priority\n"
                "- Memo discovery, warning sound, and named arrival should land as separate beats, not blended summary.\n"
                "- Each beat must state who noticed it, where they were, and what changed next.\n"
                "- Keep event order linear so room movement is easy to picture.\n\n"
            )
        if self._feedback_flag_enabled("merge_repeated_confrontation_beats") or self._feedback_mentions(
            "복도 대면",
            "밀러 접촉",
            "하나의 대화로 압축",
            "질문의 강도",
            "제자리에서 다시 시작",
        ):
            prompt += (
                "## Confrontation Compression Priority\n"
                "- If the same pair already made contact in the hallway or doorway, keep it as one continuous exchange.\n"
                "- Do not reset the scene to a fresh stare-down after the first question lands.\n"
                "- Raise pressure step by step: each next line should sharpen leverage, motive, or choice.\n\n"
            )
        if self._feedback_flag_enabled("clarify_similar_character_entries") or self._feedback_mentions("다크 수트 남자", "크리스찬 밀러", "같은 인물인지", "다른 인물인지", "헷갈"):
            prompt += (
                "## Character Entry Clarity Priority\n"
                "- If an unnamed observer later appears by name, bridge that identity once in plain prose.\n"
                "- If they are not the same person, keep appearance, role, and position cues clearly different.\n"
                "- Do not alternate between a vague label and a proper name without explaining the connection.\n\n"
            )
        if self._feedback_flag_enabled("single_axis_sentences"):
            prompt += (
                "## Single-Axis Sentence Priority\n"
                "- Keep one main beat per sentence: either movement, feeling, or judgment.\n"
                "- If a sentence starts carrying both action and analysis, split the analysis into the next cause/effect sentence.\n"
                "- Trim stock openers like '그리고', '그러자' unless the scene genuinely pivots there.\n\n"
            )
        if self._feedback_flag_enabled("single_strong_interior_beat") or self._feedback_mentions("내면 독백", "한 번만 강하게", "재진술"):
            prompt += (
                "## Interior Beat Priority\n"
                "- Keep one strong inner line for a local pressure beat, then move straight to action, dialogue, or consequence.\n"
                "- Do not paraphrase the same fear or calculation in the next sentence.\n\n"
            )
        if self._feedback_flag_enabled("avoid_metaphor_explanation") or self._feedback_mentions("비유", "은유", "의미를 다시 설명", "문단 밀도", "호흡이 무거워"):
            prompt += (
                "## Metaphor Density Priority\n"
                "- One comparison is enough. Do not follow it with a second sentence that explains the same meaning again.\n"
                "- After an image lands, return immediately to the body's reaction, dialogue, or movement.\n\n"
            )
        if self._feedback_flag_enabled("strip_meta_markers") or self._feedback_mentions(
            "메타 표식",
            "작업 메모",
            "ep01의 온도계",
            "ep01—scene21",
            "완성 원고",
        ):
            prompt += (
                "## Manuscript Cleanliness Priority\n"
                "- Do not output episode tags, scene labels, temperature notes, or work-log markers.\n"
                "- Output only finished narrative prose, never drafting metadata like 'ep01', 'scene21', or note-style labels.\n\n"
            )
        if self._feedback_flag_enabled("prefer_pivot_paragraph_breath") or self._feedback_mentions("길게 호흡", "핵심 문단"):
            prompt += (
                "## Paragraph Breath Priority\n"
                "- At one or two pivotal beats, let a fuller sentence carry the key information before the follow-up reaction.\n"
                "- Do not break every tense beat into tiny fragments when one longer cause/effect sentence reads cleaner.\n\n"
            )
        if self._feedback_mentions("처음 등장", "첫 등장", "첫 언급", "괄호", "정의", "풀어쓰기", "비유", "약어", "약자"):
            prompt += (
                "## Term Onboarding Priority\n"
                "- On first mention of a technical term/acronym, explain it only if the scene would otherwise become unclear.\n"
                "- Prefer a short inline Korean cue over parenthetical gloss or extended definition.\n"
                "- Skip analogy/metaphor unless one brief comparison is truly necessary, then return immediately to scene action.\n\n"
            )
        if self._feedback_mentions("영어 키워드", "기술 용어", "영어 표현", "해석 부담", "의미를 놓치지"):
            prompt += (
                "## Immediate Reaction After Terms\n"
                "- After any English keyword or technical term, quickly show what the character understood, feared, or physically felt.\n"
                "- Do not leave jargon hanging at sentence end without a human consequence.\n\n"
            )
        if self._feedback_mentions("동의어", "통일", "의미 중복", "혼선", "보정"):
            prompt += (
                "## Terminology Consistency Priority\n"
                "- Keep one stable term per concept; avoid alternating near-synonyms across nearby paragraphs.\n"
                "- If previous scenes already established a preferred term, keep that term unchanged.\n\n"
            )
        if self._feedback_mentions("목록", "나열", "줄바꿈", "쪼개", "분할"):
            prompt += (
                "## Dense Info Split Priority\n"
                "- Break list-like technical info into short sentences or line-broken beats.\n"
                "- Do not pack multiple condition clauses into one long sentence.\n\n"
            )
        if self._feedback_mentions("정보가 많은 단락", "정보 밀집", "요약 문장", "핵심을 정리", "핵심 정리"):
            prompt += (
                "## Dense Paragraph Summary Priority\n"
                "- End dense information paragraphs with one short summary sentence.\n"
                "- The summary line should state the immediate takeaway for scene stakes.\n\n"
            )
        if self._feedback_mentions("감각 묘사", "감각", "선명도", "1~2"):
            prompt += (
                "## Sensory Focus Priority\n"
                "- Keep sensory details focused: 1-2 sensory channels per paragraph.\n"
                "- Remove extra sensory layering that does not change scene tension.\n\n"
            )
        if self._feedback_mentions("감정의 고저", "감정 고저", "감정의 파고", "긴장 완화", "유머", "친근한 묘사"):
            prompt += (
                "## Emotional Wave Priority\n"
                "- Keep tension high overall, but insert one brief humanizing ease beat where natural.\n"
                "- Avoid flat emotional intensity across consecutive paragraphs.\n"
                "- After an easing beat, re-enter tension with a concrete trigger.\n\n"
            )
        if style == "first_person" or self._feedback_flag_enabled("force_first_person_pov"):
            prompt += (
                "## First-Person POV Priority\n"
                "- Keep Sumin's narration in first person throughout exposition and interior beats.\n"
                "- Do not refer to Sumin as '수민은/그는/그녀는' in narration unless another character is speaking.\n"
                "- Anchor perception, judgment, and immediate bodily reaction to '나' when the sentence belongs to narration.\n\n"
            )
        if self._feedback_needs_draft_cleanup():
            prompt += (
                "## Draft Cleanup Priority\n"
                "- Remove fragment-like unfinished sentences unless they are deliberate complete beats.\n"
                "- Keep names, titles, and demonstratives stable so the reader never has to guess who '그' or '그 사람' means.\n"
                "- If a sentence starts with a vague pronoun or pointer, replace it with the acting subject when needed.\n\n"
            )
        avoid_terms = sorted(self._feedback_transition_avoid_terms())
        if avoid_terms:
            prompt += (
                "## Avoid Recycled Bridge Phrases\n"
                f"- Do not use these stock transition openers unless absolutely unavoidable: {', '.join(avoid_terms[:8])}\n"
                "- Prefer concrete movement, gaze, object, or room-change cues instead.\n\n"
            )
        review_guidance = build_feedback_prompt_block(self.reader_feedback, max_items=5)
        if review_guidance:
            prompt += (
                "## Reader Feedback Priorities\n"
                "Apply these priorities while preserving the same events.\n"
                "Use them to reduce local repetition and improve reading flow.\n"
                f"{review_guidance}\n\n"
            )
        repeat_terms = self._feedback_repeat_terms()
        if repeat_terms:
            prompt += (
                "## Repetition Watch Terms\n"
                "- Reader explicitly flagged these terms as repetitive in nearby prose.\n"
                f"- Terms: {', '.join(repeat_terms[:6])}\n"
                "- Use at most one term per short paragraph unless plot-critical.\n\n"
            )
        jargon_terms = self._feedback_jargon_terms()
        if jargon_terms:
            prompt += (
                "## Jargon Watch Terms\n"
                "- Reader reported these technical terms as hard to parse when repeated.\n"
                f"- Terms: {', '.join(jargon_terms[:6])}\n"
                "- Only when needed for comprehension, add one short inline cue; later mentions should be brief callbacks.\n\n"
            )
        if previous_episode_context:
            prompt += f"## Cross-Episode Continuity\n{previous_episode_context}\n\n"

        if self.guardian_briefing:
            prompt += (
                f"## Guardian Story Briefing (일관성 유의사항)\n"
                f"아래는 Config Guardian이 분석한 이번 화의 스토리 일관성 노트입니다.\n"
                f"글을 쓸 때 이 항목들을 참고해 캐릭터 arc, 복선, 게이트 규칙을 지키세요.\n"
                f"{self.guardian_briefing}\n\n"
            )

        if beat_context:
            prompt += f"## Original Story Beats\n{beat_context}\n\n"
        if anchors:
            prompt += (
                f"## Must-Keep Evidence Anchors (use exact surface forms)\n"
                f"{anchors_text}\n\n"
            )
            prompt += (
                f"## Anchor Freshness Control\n"
                f"- Already established earlier in chapter: {recalled_text}\n"
                f"- Newly introduced in this scene: {new_anchor_text}\n"
                f"- For already established anchors, prefer one short callback only.\n"
                f"- Do not restate the same numeric claim or metric explanation more than once in this scene.\n\n"
            )
        if term_glossary:
            prompt += (
                f"## Optional Reader-Friendly Gloss (use sparingly)\n"
                f"{term_glossary}\n\n"
                f"Rule: If technical terms appear, use at most one very short inline cue when clarity truly needs it.\n\n"
            )
        if scene_character_guide:
            prompt += (
                f"## Character Voice and Visual Profiles (apply naturally)\n"
                f"{scene_character_guide}\n\n"
            )

        prompt += (
            f"{continuity}"
            f"## Task\n"
            f"Write a scene of about {word_budget} words.\n"
            f"Keep the same story events and discoveries, but render them as immersive fiction.\n"
            f"Let dialogue emerge naturally from tension and intent; avoid uniform speaking voices.\n"
            f"Keep technical exposition lightweight: do not stack unexplained jargon in consecutive sentences.\n"
            f"Only if a technical term would block comprehension, add one brief inline plain-language cue once.\n"
            f"Avoid parenthetical explanation by default, and use comparison only when clarity truly needs it.\n"
            f"For recurring concepts (e.g., coherence/drift/latency), vary wording naturally after first mention without changing meaning.\n"
            f"When reusing already-known facts, reference briefly instead of re-explaining details.\n"
            f"If a condition or responsibility point repeats, compress it into one concrete sentence and keep only the version that changes leverage or consequence.\n"
            f"Each paragraph should add one fresh observation, decision, or emotional turn instead of circling the same tension.\n"
            f"After an important statement lands, move immediately to visible consequence, reaction, or movement instead of a second paraphrase.\n"
            f"When a technical or English term appears, make the very next sentence a plain-language explanation a high-school reader can follow, then return to action or reaction.\n"
            f"Use sensory detail sparingly and only where it changes pressure, not as repeated atmosphere filler.\n"
            f"After that explanation, move straight to reaction, emotion, or decision.\n"
            f"If a scene turn already carried one body cue or inner beat, the follow-up sentence should switch to consequence, movement, or dialogue instead of repeating the same feeling.\n"
            f"If a threat, offer, or realization already landed earlier in the scene, do not restate it in new words; cash it out through consequence, interruption, or movement.\n"
            f"If a warning, alert, or denial lands, the next sentence should show a visible reaction or move instead of another explanation.\n"
            f"If the scene already moved from question to response to approach, keep that order and do not restart the earlier stage.\n"
            f"Once a room change or exit movement lands, continue from the new physical position instead of replaying the prior beat.\n"
            f"Prefer short, direct sentences over comma-heavy chains when the beat turns sharp or explanatory.\n"
            f"If the focal subject changed, establish who acts first at the paragraph opening.\n"
            f"Do not output labels, bullets, or metadata. Output only narrative prose."
        )
        prompt += (
            "\nAvoid repeating sentence openings with '그리고', '그러자', '다만', '그 직후', '잠시 뒤'."
            "\nMix sharp emphasis lines with calmer connective lines so dialogue tension is not flat."
        )
        if self._feedback_mentions("심리", "내면", "설명적", "감정선", "표정", "행동", "보여"):
            prompt += (
                "\nUse action/gesture beats to externalize emotion before adding inner analysis."
                "\nWhen a character decides, show the decision once through a small body cue or habitual motion instead of narrating the same feeling twice."
            )
        if self._feedback_mentions("장면 전환", "전환", "복도", "발표장", "흐름"):
            prompt += (
                "\nAt each location/focus change, include a short transition sentence to keep flow explicit."
                "\nIf the previous scene ended on a risk or offer, carry one concrete consequence cue into the transition instead of repeating the same warning."
            )
        if self._feedback_flag_enabled("clarify_event_transitions"):
            prompt += (
                "\nWhen a memo, alert, or named arrival shifts the scene, give it one clean sentence with location before interpretation."
            )
        if self._feedback_flag_enabled("clarify_similar_character_entries"):
            prompt += (
                "\nIf an unnamed observer becomes a named character, bridge that identity explicitly once."
            )
        if self.reader_profile.needs_role_cues():
            prompt += (
                "\n## Role Cue Priority\n"
                "- If an unnamed observer, staffer, or authority figure matters, keep one neutral role cue once and reuse it consistently.\n"
                "- Do not alternate between a vague label and a proper name unless the identity is actually clarified in the scene.\n"
                "- Give important anonymous figures one stable descriptive tag so their function stays legible.\n"
            )
        if self._feedback_flag_enabled("single_axis_sentences"):
            prompt += (
                "\nKeep one dominant beat per sentence and cut stock connective openers unless the turn truly changes there."
            )
        if self._feedback_flag_enabled("avoid_metaphor_explanation"):
            prompt += (
                "\nDo not explain a metaphor in the sentence right after it; go back to concrete action or reaction."
            )
        if self._feedback_mentions("문장 구조", "반복적인 문장 구조", "비슷한 리듬", "같은 리듬", "단조", "지루", "반복되는 표현", "묘사가 반복"):
            prompt += (
                "\nVary sentence openings and do not repeat the same clipped sentence pattern three times in a row."
            )
        if self._feedback_mentions("간결한 문장", "문맥 파악", "맥락 파악", "문맥", "맥락", "따라가기 힘들", "길고 복잡", "이해하기 어려", "이해하기 어렵"):
            prompt += (
                "\nIf a short sentence weakens context, merge it into a clearer cause-and-effect sentence."
            )
        if self._feedback_mentions("전개가 느려", "느려서 집중", "집중력을 잃", "늘어지", "템포가 느려", "속도감이 떨어"):
            prompt += (
                "\nMake sure every paragraph advances the scene instead of restating tension."
            )
        if self._feedback_mentions("말투", "어투", "톤", "대화 톤", "고유한 말투", "이해관계"):
            prompt += (
                "\nIf two speakers sound too similar, keep one line as the offer/warning and the next as the response/cost instead of repeating the same psychological point."
            )
        if self._feedback_reports_stalled_progression():
            prompt += (
                "\nIf the same pressure already landed, jump to changed consequence in the very next sentence."
            )
        if self.runtime_policy.get("prefer_scene_exit_on_stall"):
            prompt += (
                "\nReader priority: once a beat lands, prefer a concrete follow-up or scene exit over another analysis sentence."
            )
        if self._feedback_mentions("반복되는 표현", "비슷한 상황", "비슷한 상황과 묘사", "묘사가 반복", "지루"):
            prompt += (
                "\nIf the same situation has already landed, keep one sharper callback and move to consequence."
            )
        if self._feedback_mentions("긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려"):
            prompt += (
                "\nIf dialogue runs long, break it with short action/reaction beats to protect pacing."
            )
        prompt += (
            "\nWhen several supporting characters appear, give each one a different immediate pressure or motive so they stay distinct."
        )
        if self._feedback_mentions("정보가 많은 단락", "정보 밀집", "요약 문장", "핵심을 정리", "핵심 정리"):
            prompt += (
                "\nAt the end of dense information paragraphs, add one short takeaway summary sentence."
            )
        if self._feedback_mentions("감각 묘사", "감각", "선명도", "1~2"):
            prompt += (
                "\nLimit sensory focus to 1-2 sensory channels per paragraph for clarity."
            )

        # Korean long-form prose often needs a larger token budget than English
        # for the same word target; keep a higher ceiling to avoid truncation.
        scene_max_tokens = min(4800, max(1800, word_budget * 5))

        generated = self.llm.chat(
            [{"role": "user", "content": prompt}],
            system=system,
            purpose="prose_scene_gen",
            use_premium=True,
            temperature=float(self.runtime_policy.get("prose_scene_temperature", 0.75) or 0.75),
            max_tokens=scene_max_tokens,
        )
        needs_revision, reasons = self._scene_needs_readability_revision(generated)
        if needs_revision:
            generated = self._revise_scene_readability_once(
                text=generated,
                reasons=reasons,
                word_budget=word_budget,
                style=style,
            )
        return generated

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _feedback_reports_stalled_progression(self) -> bool:
        return self.reader_profile.semantic_flags.stalled_progression

    def _readability_controls(self) -> dict[str, int]:
        """Readability defaults, overridable via runtime policy."""
        min_sent = int(self.runtime_policy.get("prose_paragraph_min_sentences", 1) or 1)
        max_sent = int(self.runtime_policy.get("prose_paragraph_max_sentences", 3) or 3)
        feedback_cap = self._feedback_paragraph_sentence_cap()
        if feedback_cap is not None:
            max_sent = min(max_sent, feedback_cap)
        if self._feedback_reports_stalled_progression():
            max_sent = min(max_sent, 2)
        if self._feedback_mentions("긴 문장", "문장이 길", "긴 문단", "문단이 길", "문단", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴", "길고 복잡", "이해하기 어려", "이해하기 어렵"):
            # Reader explicitly asked for tighter paragraph breathing.
            max_sent = min(max_sent, 2)
        min_sent = max(1, min(4, min_sent))
        max_sent = max(min_sent, min(5, max_sent))
        return {"paragraph_min": min_sent, "paragraph_max": max_sent}

    def _scene_needs_readability_revision(self, text: str) -> tuple[bool, list[str]]:
        if not text or not self.reader_feedback:
            return False, []

        reasons: list[str] = []
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        max_sentences = self._readability_controls()["paragraph_max"]

        if self._feedback_mentions("긴 문장", "문장이 길", "긴 문단", "문단이 길", "문단", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴"):
            long_blocks = 0
            for block in blocks:
                sents = self._split_korean_sentences(block)
                if len(sents) > (max_sentences + 1):
                    long_blocks += 1
            if long_blocks >= 1:
                reasons.append("문단 호흡이 길고 압축이 부족함")
            dense_blocks = 0
            for block in blocks:
                acronym_hits = len(re.findall(r"\b[A-Z]{2,6}\b", block))
                paren_hits = block.count("(") + block.count(")")
                number_hits = len(re.findall(r"\d+(?:\.\d+)?", block))
                if acronym_hits + number_hits + paren_hits >= 8:
                    dense_blocks += 1
            if dense_blocks >= 1:
                reasons.append("정보가 한 문단에 과밀하게 몰려 읽기 호흡이 끊김")

        if self._feedback_mentions("기술", "기술 설명", "용어", "약자", "약어", "전문", "jargon", "acronym", "반복", "중복"):
            low = text.lower()
            jargon_terms = [
                "실시간", "보상 회로", "위상 드리프트", "t2", "t₂",
                "coherence", "drift", "latency", "qpu", "rsa-2048",
            ]
            jargon_terms.extend(self._feedback_jargon_terms())
            repeated_jargon = [t for t in jargon_terms if self._count_feedback_term_occurrences(low, t) >= 2]
            repeat_terms = self._feedback_repeat_terms()
            repeated_feedback_terms = [t for t in repeat_terms if self._count_feedback_term_occurrences(low, t) >= 2]
            if repeated_jargon:
                reasons.append(
                    "기술 용어 반복 과다: " + ", ".join(repeated_jargon[:4])
                )
            elif repeated_feedback_terms:
                reasons.append(
                    "독자 지적 반복 표현 과다: " + ", ".join(repeated_feedback_terms[:4])
                )
            if repeat_terms:
                total_repeat_hits = sum(
                    self._count_feedback_term_occurrences(low, t)
                    for t in repeat_terms[:8]
                )
                if total_repeat_hits >= 4 and not repeated_feedback_terms:
                    reasons.append("독자 지적 반복어의 누적 빈도가 높음")
            acronym_terms = re.findall(r"\b[A-Z]{2,6}\b", text)
            if len(acronym_terms) >= 6:
                reasons.append("약어/대문자 기술 표기가 과밀함")

        if self._feedback_mentions("반복", "중복", "늘어지", "묘사", "빛", "손동작", "시선", "행동 묘사"):
            repetitive_tokens = ["제스처", "표정", "손동작", "빛", "손", "시선", "어깨", "숨", "정적"]
            repeated_imagery = [t for t in repetitive_tokens if self._count_feedback_term_occurrences(text, t) >= 3]
            if repeated_imagery:
                reasons.append("유사 감각/동작 묘사 반복: " + ", ".join(repeated_imagery[:3]))
        if self._feedback_mentions("심리 표현", "비슷한 감정", "감정 표현", "내면", "반복", "중복"):
            if self._has_repetitive_emotion_phrases(text):
                reasons.append("비슷한 감정/심리 표현이 반복되어 장면 추진력이 약해짐")
        if self._feedback_mentions("조건", "책임", "통제권", "외부 지원", "대가", "의무", "권한", "계약"):
            if self._has_repeated_condition_responsibility_clauses(text):
                reasons.append("조건/책임 설명이 반복되어 장면 호흡이 늘어짐")
        if self._feedback_mentions("문장 구조", "반복적인 문장 구조", "비슷한 리듬", "같은 리듬", "단조", "지루"):
            if self._has_repetitive_sentence_openings(text):
                reasons.append("인접 문장의 시작 패턴이 반복되어 리듬이 단조로움")
            if self._has_transition_opener_streak(text):
                reasons.append("'그리고/그러자/다만' 계열 연결어 시작이 반복되어 흐름이 뻣뻣함")
        if self._feedback_flag_enabled("avoid_metaphor_explanation") or self._feedback_mentions("비유", "은유", "의미를 다시 설명", "문단 밀도", "호흡이 무거워"):
            if self._has_post_metaphor_explanation_pairs(text):
                reasons.append("비유 직후 의미를 다시 풀어 문단 밀도가 높아짐")
        if self._feedback_mentions("쉼표", "연결어", "쉼표와 접속", "문장이 너무 길", "길고 복잡", "호흡", "걸리는"):
            comma_heavy = 0
            for sent in self._split_korean_sentences(text):
                if (sent.count(",") + sent.count("，") + sent.count(";")) >= 2:
                    comma_heavy += 1
                    continue
                if len(re.findall(r"(그리고|그러나|하지만|다만|또한|한편|그래서|그러자)", sent)) >= 2:
                    comma_heavy += 1
            if comma_heavy >= 2:
                reasons.append("쉼표/접속으로 이어지는 문장이 많아 호흡이 걸림")
        if self._feedback_mentions("동의어", "통일", "의미 중복", "혼선"):
            synonym_groups = [
                ("보정", "실시간", "드리프트"),
            ]
            for group in synonym_groups:
                hits = sum(1 for token in group if self._count_feedback_term_occurrences(text, token) >= 1)
                if hits >= 2:
                    reasons.append("같은 개념의 용어가 혼용되어 일관성이 약함")
                    break
        if self._feedback_mentions("정보가 많은 단락", "정보 밀집", "요약 문장", "핵심을 정리", "핵심 정리"):
            dense_without_summary = 0
            for block in blocks:
                sent_list = self._split_korean_sentences(block)
                if len(sent_list) < 2:
                    continue
                dense_hits = (
                    len(re.findall(r"\d+(?:\.\d+)?", block))
                    + len(re.findall(r"\b[A-Z]{2,6}\b", block))
                    + block.count("(")
                )
                if dense_hits < 4:
                    continue
                tail = sent_list[-1]
                if not re.search(r"(요약|정리|결국|핵심|한마디로|즉)", tail):
                    dense_without_summary += 1
            if dense_without_summary >= 1:
                reasons.append("정보 밀집 단락 말미의 핵심 요약 문장이 부족함")
        if self._feedback_mentions("가능성", "계산", "추론", "판단", "심리", "내면", "반복", "중복", "늘어지"):
            if self._has_repetitive_cognitive_terms(text):
                reasons.append("심리 추론 어휘가 반복되어 긴장 템포가 느려짐")
        if self._feedback_mentions("조건", "책임", "통제권", "외부 지원", "대가", "의무", "권한", "계약"):
            if self._has_repeated_condition_responsibility_clauses(text):
                reasons.append("조건/책임 설명이 반복되어 장면 호흡이 늘어짐")
        if self._feedback_mentions("간결한 문장", "문맥 파악", "맥락 파악", "문맥", "맥락", "따라가기 힘들"):
            if self._has_excess_clipped_sentences(text):
                reasons.append("짧은 문장이 누적되어 문맥 연결이 자주 끊김")
        if self._feedback_reports_stalled_progression() or self._feedback_mentions("전개가 느려", "느려서 집중", "집중력을 잃", "늘어지", "템포가 느려", "속도감이 떨어"):
            if self._has_low_momentum_paragraphs(text):
                reasons.append("상황 진전 없이 같은 압박을 반복하는 문단이 있어 전개가 느림")
        if self._feedback_mentions("인물", "역할", "의도", "설명", "템포", "느려"):
            role_explain_blocks = 0
            for block in blocks:
                explain_hits = len(re.findall(r"(역할|의도|담당|정체|소개|설명|관계)", block))
                if explain_hits >= 3:
                    role_explain_blocks += 1
            if role_explain_blocks >= 1:
                reasons.append("인물 역할/의도 설명이 과밀해 전개 속도가 느림")
        if self._feedback_needs_draft_cleanup():
            if re.search(r"\b(real[- ]time viable if externally supported)\b", text, re.IGNORECASE):
                reasons.append("영어 초안 흔적이 남아 완성 원고처럼 읽히지 않음")
            fragment_runs = 0
            for block in blocks:
                clipped = [
                    sent for sent in self._split_korean_sentences(block)
                    if len(re.findall(r"[0-9A-Za-z가-힣]+", sent)) <= 4
                    and not re.search(r"[\"“”'‘’]", sent)
                ]
                if len(clipped) >= 2:
                    fragment_runs += 1
            if fragment_runs >= 1:
                reasons.append("미완·초안처럼 보이는 짧은 문장 뭉침이 남아 있음")

        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "인물", "역할", "구분", "호칭", "이름", "말투", "어투", "톤", "speaker"):
            ambiguous_dialogue_lines = 0
            quote_lines = re.findall(r"[^\n]*[\"“][^\"”\n]{3,}[\"”][^\n]*", text)
            cue_pattern = re.compile(r"(?:[가-힣A-Za-z]{2,}\s*(?:가|이|은|는|을|를)|씨|님|교수|선배|요원|박사|사장|대표|말했|물었|받았|응답했)")
            for line in quote_lines:
                if not cue_pattern.search(line):
                    ambiguous_dialogue_lines += 1
            if ambiguous_dialogue_lines >= 2:
                reasons.append("화자 단서 부족한 대사 라인이 연속됨")
            if self._dialogue_voice_is_monotone(text):
                reasons.append("대사 말투/문장 종결 패턴이 단조로워 화자 구분이 약함")
            if self._dialogue_function_is_blurred(text):
                reasons.append("서로 다른 화자의 제안/경고 기능이 겹쳐 역할 구분이 흐려짐")
        if self._feedback_mentions("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트"):
            if self._has_expository_dialogue_cluster(text):
                reasons.append("정보 전달형 대사가 길어 감정 임팩트가 약해짐")
        if self._feedback_mentions("긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려"):
            if self._has_expository_dialogue_cluster(text):
                reasons.append("긴 대화 구간이 길게 이어져 장면 템포가 느려짐")
            quote_blocks = sum(1 for b in blocks if len(re.findall(r"[\"“][^\"”\n]{6,}[\"”]", b)) >= 2)
            if quote_blocks >= 2:
                reasons.append("연속 대사 비중이 높아 액션/반응 비트가 부족함")

        return bool(reasons), reasons

    def _revise_scene_readability_once(
        self,
        text: str,
        reasons: list[str],
        word_budget: int,
        style: str,
    ) -> str:
        style = self._effective_style(style)
        pov = "first person" if style == "first_person" else "third person close"
        reason_text = "; ".join(r for r in reasons if r).strip() or "가독성 개선 필요"
        sentence_cap = self._feedback_sentence_word_cap(default=25)
        first_person_cleanup_line = (
            "- 수민 서술은 1인칭으로 유지하고 내레이션에서 '수민은/그는' 식 자기지칭을 지울 것\n"
            if style == "first_person" else ""
        )
        draft_cleanup_line = (
            "- 미완 문장, 대명사 흔들림, 호칭·지시어 혼선을 정리해 완성 원고처럼 읽히게 할 것\n"
            if self._feedback_needs_draft_cleanup() else ""
        )
        prompt = (
            "다음 한국어 장면 산문을 같은 사건 흐름으로 유지하면서 1회 리라이트하라.\n"
            f"개선 사유: {reason_text}\n\n"
            "제약:\n"
            f"- 시점 유지: {pov}\n"
            f"- 분량 유지: 약 {word_budget}단어(크게 벗어나지 말 것)\n"
            "- 긴 설명문을 줄이고 문단 호흡을 짧게 분할\n"
            f"- 대부분의 문장은 약 {sentence_cap}어절 이하로 유지하고, 인과가 길어지면 둘로 나눌 것\n"
            "- 같은 기술 용어/수치를 연속 문단에서 반복 설명하지 말 것\n"
            "- 기술 용어 첫 언급만 짧게 풀고 이후는 짧은 콜백으로 처리\n"
            "- 약어/대문자 기술 표기는 첫 등장에만 짧게 풀고 이후 최소화\n"
            "- 가능성/계산/추론 같은 내면 분석 어휘는 반복하지 말고 행동으로 치환\n"
            "- 이미 등장한 인물의 역할/의도 재설명은 축약하고 장면 진행을 우선\n"
            "- 같은 문장 시작 패턴이나 단문 리듬을 3회 이상 반복하지 말 것\n"
            "- '그리고', '그러자', '다만' 같은 연결어 시작은 근접 문장에서 반복하지 말 것\n"
            "- 짧은 문장은 앞뒤 문장과 인과관계가 분명할 때만 단독으로 둘 것\n"
            "- 같은 박자의 단문이 2개 이상 이어지면 1개의 자연스러운 복합문으로 묶을 것\n"
            "- 강하게 압박하는 문장과 담백하게 상황을 잇는 문장을 섞어 리듬 고저를 만들 것\n"
            "- 같은 상황이나 감정을 다른 말로 되풀이하지 말고, 이미 성립한 내용은 결과만 짧게 남길 것\n"
            "- 비슷한 감각 묘사와 심리 표현은 한 번만 선명하게 쓰고, 나머지는 행동/결과로 압축할 것\n"
            "- 조건, 책임, 통제권, 대가 같은 말이 반복되면 하나의 구체적 문장으로 묶고 나머지는 결과로 넘길 것\n"
            "- 위협이 추상적이면 사람, 물건, 규정, 공간 반응 중 하나로 한 번만 고정하고 같은 불안을 반복 설명하지 말 것\n"
            "- 이름이 없는 중요한 인물은 한 번만 중립적 역할 단서로 고정하고 이후에는 같은 단서를 일관되게 사용할 것\n"
            "- 한 문장에는 동작, 감정, 판단 가운데 한 축만 남기고 둘 이상이면 인과관계가 보이게 분리할 것\n"
            "- 비유나 비교를 쓴 직후 그 의미를 다음 문장으로 다시 해설하지 말 것\n"
            "- 이미 장면에 깔린 위협, 제안, 깨달음은 새 말로 되풀이하지 말고 결과, 끼어듦, 이동으로 처리할 것\n"
            "- 각 문단은 반드시 상황 변화, 압박, 발견 중 하나를 전진시킬 것\n"
            "- 어려운 기술 개념은 필요할 때만 짧은 일상 비유나 은유를 붙이고 바로 행동으로 돌아갈 것\n"
            "- 영어 키워드나 기술 용어 뒤에는 다음 문장으로 쉬운 풀어쓰기를 한 번 붙인 뒤, 곧바로 인물의 즉각적 반응, 감정, 판단을 붙일 것\n"
            "- 경보, 접근 거부, 알림이 나오면 다음 문장은 해석이 아니라 바로 보이는 반응이나 움직임으로 넘길 것\n"
            "- 쉼표와 접속으로 절이 길게 이어지면 짧고 분명한 두 문장으로 나눌 것\n"
            "- 사건, 발견, 감정선의 순서는 바꾸지 말 것\n"
            f"{first_person_cleanup_line}"
            f"{draft_cleanup_line}"
            "- 출력은 소설 본문만\n\n"
            f"원문:\n{text}"
        )
        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "인물", "역할", "구분", "호칭", "이름", "말투", "어투", "톤", "speaker"):
            prompt += (
                "\n추가 제약:\n"
                "- 대사 구간은 화자/청자 단서를 자주 배치해 혼선을 줄일 것\n"
                "- 동일 인물은 인접 문단에서 호칭을 과도하게 바꾸지 말 것\n"
            )
        if self.reader_profile.needs_role_cues():
            prompt += (
                "- 이름이 없는 관찰자/조력자/권위자는 한 번만 중립적 역할 단서로 고정하고 이후에는 같은 단서를 유지할 것\n"
            )
        if self._feedback_mentions("말투", "어투", "톤", "대화 톤", "고유한 말투"):
            prompt += (
                "- 인접한 대사는 문장 종결 어미/리듬을 분산해 화자별 말투 차이를 유지할 것\n"
            )
        if self._feedback_mentions("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트"):
            prompt += (
                "- 정보 전달형 대사는 한 문장으로 압축하고, 바로 반응/행동 단서를 붙일 것\n"
                "- 한 대사 안에서 설명 접속어를 연쇄적으로 사용하지 말 것\n"
            )
        if self._feedback_mentions("긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려"):
            prompt += (
                "- 긴 대화 구간은 1-2회 발화마다 짧은 행동/환경 반응 비트를 삽입할 것\n"
                "- 설명형 대사를 연속으로 길게 배치하지 말 것\n"
            )
        if self._feedback_mentions("문장 구조", "반복적인 문장 구조", "비슷한 리듬", "같은 리듬", "단조", "지루", "반복되는 표현", "묘사가 반복"):
            prompt += (
                "- 인접 문장에서 주어 시작과 종결 리듬을 분산해 단조로운 문장 구조 반복을 피할 것\n"
            )
        if self._feedback_mentions("간결한 문장", "문맥 파악", "맥락 파악", "문맥", "맥락", "따라가기 힘들", "길고 복잡", "이해하기 어려", "이해하기 어렵"):
            prompt += (
                "- 지나치게 짧은 문장이 이어지면 1개 이상을 연결해 누가/왜/어디서가 드러나게 만들 것\n"
            )
        if self._feedback_mentions("전개가 느려", "느려서 집중", "집중력을 잃", "늘어지", "템포가 느려", "속도감이 떨어"):
            prompt += (
                "- 이미 성립한 긴장이나 정보를 반복 설명하지 말고 바로 다음 반응 또는 결정으로 넘어갈 것\n"
            )
        if self._feedback_mentions("반복되는 표현", "비슷한 상황", "비슷한 상황과 묘사", "묘사가 반복", "지루"):
            prompt += (
                "- 비슷한 상황과 묘사를 다른 표현으로 반복하지 말고, 장면이 달라진 지점만 남길 것\n"
            )
        if self._feedback_mentions("감정의 고저", "감정 고저", "감정의 파고", "긴장 완화", "유머", "친근한 묘사"):
            prompt += (
                "- 감정 강도를 평탄하게 유지하지 말고, 짧은 완화 비트 후 다시 긴장을 세울 것\n"
                "- 완화 비트는 작은 인간적 반응(짧은 농담, 숨 고르기, 주변 감각) 수준으로 제한할 것\n"
            )
        revised = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_scene_readability_revise",
            use_premium=True,
            temperature=float(self.runtime_policy.get("prose_scene_readability_temperature", 0.35) or 0.35),
            max_tokens=min(4800, max(1800, word_budget * 5)),
        )
        return revised or text

    @staticmethod
    def _dialogue_voice_is_monotone(text: str) -> bool:
        """
        Detect low variety in quoted dialogue endings, a common source of speaker blur.
        """
        quoted = re.findall(r"[\"“]([^\"”\n]{3,120})[\"”]", text or "")
        if len(quoted) < 4:
            return False
        endings: list[str] = []
        for q in quoted[:10]:
            q = re.sub(r"\s+", " ", q).strip()
            if not q:
                continue
            endings.append(q[-2:] if len(q) >= 2 else q)
        if len(endings) < 4:
            return False
        unique = len(set(endings))
        return unique <= 2

    @staticmethod
    def _dialogue_function_is_blurred(text: str) -> bool:
        quoted = re.findall(r"[\"“]([^\"”\n]{3,140})[\"”]", text or "")
        if len(quoted) < 4:
            return False
        normalized: list[str] = []
        for q in quoted[:8]:
            q = re.sub(r"\s+", " ", q).strip()
            if not q:
                continue
            q = re.sub(r"^(?:나는|우리는|저는|그는|그녀는|수민은|밀러는|모레노는)\s+", "", q)
            q = re.sub(r"(?:라고|하며|물으며|답하며|덧붙이며).*$", "", q)
            normalized.append(q[:40].lower())
        if len(normalized) < 4:
            return False
        return len(set(normalized)) <= max(2, len(normalized) - 2)

    def _feedback_mentions(self, *keywords: str) -> bool:
        return self.reader_profile.mentions(*keywords)

    def _feedback_repeat_terms(self) -> list[str]:
        return self.reader_profile.repeat_terms(max_terms=10)

    def _feedback_jargon_terms(self) -> list[str]:
        return self.reader_profile.jargon_terms(max_terms=10)

    def _feedback_style_constraints(self) -> dict:
        return self.reader_profile.style_constraints()

    def _effective_style(self, style: str) -> str:
        requested = str(style or "third_person_close").strip() or "third_person_close"
        if requested != "first_person" and self.reader_profile.flag_enabled("force_first_person_pov"):
            return "first_person"
        return requested

    def _feedback_flag_enabled(self, key: str, default: bool = False) -> bool:
        return self.reader_profile.flag_enabled(key, default=default)

    def _feedback_term_repeat_cap(self, default: int = 2) -> int:
        return self.reader_profile.term_repeat_cap(default=default)

    def _feedback_sentence_word_cap(self, default: int = 25) -> int:
        return self.reader_profile.sentence_word_cap(default=default)

    def _feedback_paragraph_sentence_cap(self) -> Optional[int]:
        return self.reader_profile.paragraph_sentence_cap()

    def _feedback_dense_sentence_cap(self, default: int = 2) -> int:
        return self.reader_profile.dense_sentence_cap(default=default)

    def _feedback_jargon_term_cap(self, default: int = 2) -> int:
        return self.reader_profile.jargon_term_cap(default=default)

    def _feedback_sensory_channel_cap(self, default: int = 2) -> int:
        return self.reader_profile.sensory_channel_cap(default=default)

    def _feedback_emotion_repeat_cap(self, default: int = 1) -> int:
        return self.reader_profile.emotion_repeat_cap(default=default)

    def _feedback_transition_char_window(self) -> tuple[int, int]:
        return self.reader_profile.transition_char_window()

    def _feedback_short_beat_char_window(self) -> tuple[int, int]:
        return self.reader_profile.short_beat_char_window()

    def _feedback_short_beats_per_scene(self) -> tuple[int, int]:
        return self.reader_profile.short_beats_per_scene()

    def _feedback_transition_opener_cap(self, default: int = 2) -> int:
        return self.reader_profile.transition_opener_cap(default=default)

    def _feedback_transition_avoid_terms(self) -> set[str]:
        return self.reader_profile.transition_avoid_terms()

    def _feedback_sentence_variety_window(self, default: int = 4) -> int:
        return self.reader_profile.sentence_variety_window(default=default)

    def _feedback_needs_draft_cleanup(self) -> bool:
        return self.reader_profile.needs_draft_cleanup()

    @staticmethod
    def _count_feedback_term_occurrences(text: str, term: str) -> int:
        return count_feedback_term_occurrences(text, term)

    def _reader_feedback_final_pass(
        self,
        text: str,
        target_words: int,
        style: str,
        chapter_anchors: Optional[list[str]] = None,
    ) -> str:
        # Compatibility shim: chapter cleanup lives in ChapterPolisher now.
        return self.chapter_polisher.apply_reader_feedback_pass(
            text,
            target_words,
            style,
            chapter_anchors,
            prose_adapter=self,
        )

    @staticmethod
    def _has_repetitive_cognitive_terms(text: str) -> bool:
        """
        Detect repeated analytic-inner wording that readers experience as drag.
        """
        raw = str(text or "").lower()
        if not raw:
            return False
        cues = [
            "가능성", "계산", "판단", "추론", "결론", "가정", "시나리오", "확률",
        ]
        repeated_types = sum(1 for cue in cues if raw.count(cue) >= 2)
        return repeated_types >= 1

    @staticmethod
    def _has_repeated_condition_responsibility_clauses(text: str) -> bool:
        sentences = [
            re.sub(r"\s+", " ", sent).strip()
            for sent in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(text or ""))
            if sent and str(sent).strip()
        ]
        if len(sentences) < 2:
            return False
        pattern = re.compile(r"(조건|책임|통제권|외부 지원|대가|의무|권한|계약|부담)")
        hits = [sent for sent in sentences if pattern.search(sent)]
        if len(hits) < 2:
            return False
        overlap = 0
        prev_terms: set[str] = set()
        for sent in hits:
            terms = set(pattern.findall(sent))
            if prev_terms & terms:
                overlap += 1
            prev_terms = terms
        return overlap >= 1 or len(hits) >= 3

    def _has_repetitive_emotion_phrases(self, text: str) -> bool:
        sentences = self._split_korean_sentences(text)
        if len(sentences) < 4:
            return False
        prev_sig = ""
        streak = 0
        for sent in sentences:
            sig = self._emotion_signature(sent)
            if not sig:
                continue
            if sig == prev_sig:
                streak += 1
                if streak >= max(1, self._feedback_emotion_repeat_cap(default=1)):
                    return True
            else:
                prev_sig = sig
                streak = 0
        return False

    def _has_repetitive_sentence_openings(self, text: str) -> bool:
        sentences = self._split_korean_sentences(text)
        openers: list[str] = []
        for sent in sentences:
            cleaned = re.sub(r"^[\"“”'‘’\(\)\[\]\s]+", "", str(sent or "").strip())
            match = re.match(r"([0-9A-Za-z가-힣]{1,8})", cleaned)
            if not match:
                continue
            openers.append(match.group(1).lower())
        if len(openers) < 5:
            return False
        streak = 1
        for idx in range(1, len(openers)):
            if openers[idx] == openers[idx - 1]:
                streak += 1
                if streak >= 3:
                    return True
            else:
                streak = 1
        sample = openers[:8]
        return len(set(sample)) <= max(2, len(sample) // 3)

    @staticmethod
    def _sentence_leading_connector(sentence: str) -> str:
        cleaned = re.sub(r"^[\"“”'‘’\(\)\[\]\s]+", "", str(sentence or "").strip())
        match = re.match(r"(그리고|그러자|다만|하지만|그러나|한편|곧이어|이어서|그 순간)\b", cleaned)
        return match.group(1) if match else ""

    def _has_transition_opener_streak(self, text: str) -> bool:
        cap = self._feedback_transition_opener_cap(default=2)
        connectors = [
            self._sentence_leading_connector(sent)
            for sent in self._split_korean_sentences(text)
        ]
        connectors = [c for c in connectors if c in {"그리고", "그러자", "다만", "이어서", "그 순간"}]
        if len(connectors) < max(2, cap + 1):
            return False
        streak = 1
        for idx in range(1, len(connectors)):
            if connectors[idx] == connectors[idx - 1]:
                streak += 1
                if streak >= cap + 1:
                    return True
            else:
                streak = 1
        return any(connectors.count(conn) >= cap + 1 for conn in {"그리고", "그러자", "다만", "이어서", "그 순간"})

    def detect_repetition_pattern_warnings(self, text: str) -> list[str]:
        if not text:
            return []
        warnings: list[str] = []
        cap = self._feedback_transition_opener_cap(default=2)
        connector_counts: dict[str, int] = {"그리고": 0, "그러자": 0, "다만": 0, "이어서": 0, "그 순간": 0}
        for sent in self._split_korean_sentences(text):
            conn = self._sentence_leading_connector(sent)
            if conn in connector_counts:
                connector_counts[conn] += 1
        overused_connectors = [
            f"{conn} {count}회"
            for conn, count in connector_counts.items()
            if count >= cap + 1
        ]
        if overused_connectors:
            warnings.append("연결어 반복 감지: " + ", ".join(overused_connectors))
        if self._has_transition_opener_streak(text):
            warnings.append("인접 문장에서 같은 연결어 시작 패턴이 반복됨")
        if self._has_repetitive_sentence_openings(text):
            warnings.append("인접 문장 시작 패턴이 반복됨")

        repeated_patterns: list[str] = []
        seen_fp: dict[str, int] = {}
        for sent in self._split_korean_sentences(text):
            fp = self._sentence_fingerprint(sent)
            if len(fp.split()) < 3:
                continue
            seen_fp[fp] = seen_fp.get(fp, 0) + 1
            if seen_fp[fp] == 2:
                repeated_patterns.append(self._truncate_text(re.sub(r"\s+", " ", sent).strip(), 24))
        if repeated_patterns:
            warnings.append("유사 문장 패턴 반복: " + ", ".join(repeated_patterns[:3]))
        if self._has_monotone_sentence_length_run(text):
            warnings.append("강한 문장과 담백한 문장의 길이 대비가 부족함")
        return warnings

    def _has_monotone_sentence_length_run(self, text: str) -> bool:
        sentences = self._split_korean_sentences(text)
        if len(sentences) < 5:
            return False
        bands: list[str] = []
        for sent in sentences:
            wc = self._sentence_word_count(sent)
            if wc <= 6:
                bands.append("short")
            elif wc <= 16:
                bands.append("medium")
            else:
                bands.append("long")
        streak = 1
        for idx in range(1, len(bands)):
            if bands[idx] == bands[idx - 1]:
                streak += 1
                if streak >= 4:
                    return True
            else:
                streak = 1
        return False

    def _has_excess_clipped_sentences(self, text: str) -> bool:
        sentences = self._split_korean_sentences(text)
        if len(sentences) < 6:
            return False
        short_count = 0
        short_streak = 0
        for sent in sentences:
            if self._sentence_word_count(sent) <= 5:
                short_count += 1
                short_streak += 1
                if short_streak >= 3:
                    return True
            else:
                short_streak = 0
        return short_count >= 4 and (short_count / max(len(sentences), 1)) >= 0.35

    def _has_low_momentum_paragraphs(self, text: str) -> bool:
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        static_blocks = 0
        for block in blocks:
            sentences = self._split_korean_sentences(block)
            if len(sentences) < 2:
                continue
            low = block.lower()
            change_hits = len(re.findall(r"(결국|곧|바로|그때|마침내|곧장|이어|그러자|하지만|그 순간)", block))
            action_hits = len(re.findall(r"(움직|돌렸|건넸|밀었|열었|접었|멈췄|들었|내려놓|올려다|바뀌|흔들|숨을)", block))
            repeat_hits = len(re.findall(r"(생각|계산|판단|설명|반복|다시|정리하면|즉|의미는|뜻이었다)", low))
            pressure_hits = len(re.findall(r"(정적|침묵|긴장|압박|불안|숨을 죽|시선이 멈|말끝이 무거|턱선이 굳)", low))
            quote_hits = len(re.findall(r"[\"“][^\"”\n]{8,}[\"”]", block))
            if change_hits == 0 and action_hits <= 1 and (repeat_hits >= 2 or pressure_hits >= 2):
                static_blocks += 1
                continue
            if quote_hits >= 2 and change_hits == 0 and action_hits == 0 and repeat_hits >= 1:
                static_blocks += 1
        return static_blocks >= 1

    @staticmethod
    def _has_expository_dialogue_cluster(text: str) -> bool:
        """
        Detect long quote clusters that read like pure information dumps.
        """
        quoted = re.findall(r"[\"“]([^\"”\n]{8,220})[\"”]", text or "")
        if not quoted:
            return False
        explain_pattern = re.compile(r"(왜냐하면|즉|다시 말해|정리하면|요약하면|핵심은|결론은|설명하자면)")
        flagged = 0
        for q in quoted:
            low = q.lower()
            explain_hits = len(explain_pattern.findall(low))
            technical_hits = len(re.findall(r"\b[A-Z]{2,8}\b", q)) + len(re.findall(r"\d+(?:\.\d+)?", q))
            if len(q) >= 90 and (explain_hits >= 1 or technical_hits >= 2):
                flagged += 1
            elif explain_hits >= 2:
                flagged += 1
        return flagged >= 1

    def _build_scene_term_glossary(
        self,
        scene: DistilledScene,
        matched_beats: list[tuple[str, str]],
        blocked_terms: Optional[list[str]] = None,
    ) -> str:
        """
        Build short plain-language gloss hints for frequent technical terms.
        Keeps jargon readable without flattening techno-thriller tone.
        """
        if not bool(self.runtime_policy.get("prose_enable_term_gloss", True)):
            return ""

        source = "\n".join(
            [d for d in scene.discoveries if isinstance(d, str)]
            + [btxt for _, btxt in matched_beats if btxt]
            + [scene.narrative_summary or ""]
        )
        if not source.strip():
            return ""

        glossary = {
            "QPU": "양자 계산을 실제로 처리하는 칩",
            "RSA-2048": "일반적으로 깨기 어려운 암호 체계",
            "DARPA": "미국 국방 고등연구 프로젝트 기관",
            "NSA": "미국 국가안보 관련 정보기관",
            "50ms": "눈 깜빡임에 가까운 아주 짧은 지연",
            "T₂": "양자 상태가 버티는 시간 척도",
            "latency": "입력부터 반응까지 걸리는 시간",
            "protocol": "시스템이 합의해 따르는 규칙",
            "fail-safe": "문제가 나면 자동으로 안전 모드로 전환하는 장치",
        }

        hits: list[str] = []
        low_source = source.lower()
        blocked = {str(t).lower() for t in (blocked_terms or [])}
        for term in glossary:
            if term.lower() in low_source:
                if any(term.lower() in b or b in term.lower() for b in blocked):
                    continue
                hits.append(f"- {term}: {glossary[term]}")
        for term in self._feedback_jargon_terms():
            low_term = term.lower()
            if low_term not in low_source:
                continue
            if any(low_term in b or b in low_term for b in blocked):
                continue
            if any(low_term in h.lower() for h in hits):
                continue
            # Keep gloss minimal when term-specific dictionary entry does not exist.
            hits.append(f"- {term}: 처음 한 번만 짧은 쉬운 말로 풀어 제시")

        return "\n".join(hits[:6])

    @staticmethod
    def _extract_anchor_terms(text: str) -> list[str]:
        """
        Extract concrete terms worth preserving verbatim in prose.
        Focus: money, protocol IDs, all-caps codes, mixed alpha-num tags, times.
        """
        if not text:
            return []
        pats = [
            r"\$[0-9][0-9,]*",
            r"[A-Z]{2,}(?:-[A-Z0-9]{2,})+",
            r"[A-Z]{2,}[0-9]{2,}",
            r"\b(?:Phase-Guard|PH-GRD|Greyshore|Benefactor|NSA|DARPA|LST|QPU|RSA-2048)\b",
            r"(?:월요일\s*자정|자정|항만)",
            r"[0-9]{3,}",
        ]
        out: list[str] = []
        for pat in pats:
            out.extend(re.findall(pat, text, flags=re.IGNORECASE))
        # normalize/dedupe preserve order
        seen = set()
        uniq: list[str] = []
        for t in out:
            s = t.strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(s)
        return uniq

    def _collect_episode_anchor_terms(self, episode_context: dict) -> list[str]:
        """Collect anchors that should survive across the full chapter."""
        raw_parts: list[str] = [str(episode_context.get("summary", ""))]
        for beat in episode_context.get("beats", []):
            if isinstance(beat, dict):
                raw_parts.append(str(beat.get("content", "")))
        source = "\n".join(raw_parts)
        anchors = self._extract_anchor_terms(source)

        # Also keep short quoted phrases from episode config (often mission-critical).
        quote_terms = re.findall(r"[\"“”']([^\"“”']{4,40})[\"“”']", source)
        for q in quote_terms:
            q = q.strip()
            if q and q not in anchors:
                anchors.append(q)
        return anchors[:40]

    @staticmethod
    def _select_anchor_terms_for_coverage(anchors: list[str]) -> list[str]:
        """
        Keep only high-signal anchors for final coverage checks.
        Avoid forcing dense numeric jargon that often causes repetitive restatement.
        """
        if not anchors:
            return []

        primary: list[str] = []
        fallback: list[str] = []
        seen: set[str] = set()

        for raw in anchors:
            term = str(raw or "").strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)

            # Skip pure numeric tokens; they are often redundant in prose.
            if re.fullmatch(r"[0-9]{3,}", term):
                continue
            if len(term) < 3:
                continue

            has_letters = bool(re.search(r"[A-Za-z가-힣]", term))
            has_caps_code = bool(re.search(r"[A-Z]{2,}", term))
            has_hyphen = "-" in term

            if has_letters and (has_caps_code or has_hyphen):
                primary.append(term)
            elif has_letters and len(term) <= 28:
                fallback.append(term)

        selected = (primary + fallback)[:12]
        return selected

    def _tune_coverage_anchors(self, anchors: list[str]) -> list[str]:
        """
        Reader feedback often flags technical-term density and repetition.
        Keep enough anchors for evidence fidelity, but avoid jargon-heavy overload.
        """
        if not anchors:
            return []
        strict_jargon_control = self.reader_profile.prefers_technical_term_restraint()
        max_terms = 8 if strict_jargon_control else 12
        tuned: list[str] = []
        for term in anchors:
            if strict_jargon_control and self._is_dense_jargon_anchor(term):
                continue
            tuned.append(term)
            if len(tuned) >= max_terms:
                break
        if tuned:
            return tuned
        return anchors[: max(4, min(max_terms, len(anchors)))]

    @staticmethod
    def _is_dense_jargon_anchor(term: str) -> bool:
        t = str(term or "").strip()
        if not t:
            return False
        has_caps = bool(re.search(r"[A-Z]{2,}", t))
        has_num = bool(re.search(r"\d", t))
        has_symbol = bool(re.search(r"[-_/×%]", t))
        has_caps_only = bool(re.fullmatch(r"[A-Z]{2,8}", t))
        return (has_caps and has_num) or (has_caps and has_symbol) or has_caps_only

    # ------------------------------------------------------------------ #
    # Transitions
    # ------------------------------------------------------------------ #

    def _combine_with_transitions(
        self,
        sections: list[str],
        scenes: list[DistilledScene],
        episode_context: dict,
        style: str,
    ) -> str:
        """Combine prose sections with internal monologue transitions."""
        if len(sections) <= 1:
            return sections[0] if sections else ""

        style = self._effective_style(style)
        pov = "first person" if style == "first_person" else "third person close"
        parts = [sections[0]]

        for i in range(1, len(sections)):
            prev_scene = scenes[i - 1] if i - 1 < len(scenes) else None
            next_scene = scenes[i] if i < len(scenes) else None

            bridge = self._generate_transition(
                prev_tail=sections[i - 1][-300:],
                next_head=sections[i][:300],
                prev_scene=prev_scene,
                next_scene=next_scene,
                pov=pov,
            )
            bridge = self._ensure_transition_marker(bridge)
            parts.append(bridge)
            parts.append(sections[i])

        return "\n\n".join(parts)

    def _generate_transition(
        self,
        prev_tail: str,
        next_head: str,
        prev_scene: Optional[DistilledScene],
        next_scene: Optional[DistilledScene],
        pov: str,
    ) -> str:
        """Generate a 2-4 sentence transition between scenes."""
        prev_title = prev_scene.title if prev_scene else "previous scene"
        next_title = next_scene.title if next_scene else "next scene"
        prev_loc = (prev_scene.location if prev_scene and prev_scene.location else "unknown")
        next_loc = (next_scene.location if next_scene and next_scene.location else "unknown")
        next_chars = ", ".join((next_scene.characters_present if next_scene else [])[:4]) or "unknown"

        prompt = (
            f"Write a 2-4 sentence {pov} brief scene transition.\n\n"
            f"Leaving: '{prev_title}'\n"
            f"Leaving location: {prev_loc}\n"
            f"...{prev_tail}\n\n"
            f"Entering: '{next_title}'\n"
            f"Entering location: {next_loc}\n"
            f"Entering characters (important): {next_chars}\n"
            f"{next_head}...\n\n"
            f"The transition should feel like a natural breath between moments: "
            f"one concrete movement + one short thought + one attention shift. "
            f"Make spatial continuity explicit in one sentence, and avoid abstract wording. "
            f"{'The bridge must include one concrete object, doorway, or room-state cue instead of a vague connective beat. ' if self.runtime_policy.get('prefer_concrete_transition_cue') else ''}"
            f"Vary the bridge phrasing; do not reuse the same stock opener as the previous transition. "
            f"Do not begin with stock time adverbs like '그 직후' or '잠시 뒤'; "
            f"use movement, gaze, doorway, hallway, or object handling instead. "
            f"Write in Korean. Write ONLY the transition text."
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_transition",
            use_premium=True,
            temperature=float(self.runtime_policy.get("prose_transition_temperature", 0.7) or 0.7),
            max_tokens=200,
        )

    # ------------------------------------------------------------------ #
    # Polish
    # ------------------------------------------------------------------ #

    def _polish(
        self,
        text: str,
        target_words: int,
        style: str,
        chapter_anchors: Optional[list[str]] = None,
    ) -> str:
        # Compatibility shim: prose generation delegates late-stage polish ownership.
        return self.chapter_polisher.run_llm_polish(
            text,
            target_words,
            style,
            chapter_anchors,
            prose_adapter=self,
        )

    def _ensure_anchor_coverage(
        self,
        text: str,
        chapter_anchors: list[str],
        target_words: int,
        style: str,
    ) -> str:
        return self.chapter_polisher.ensure_anchor_coverage(
            text,
            chapter_anchors,
            target_words,
            style,
            prose_adapter=self,
        )

    def _enforce_pov_timeline_guards(
        self,
        text: str,
        style: str,
        protagonist_name: str,
    ) -> str:
        """
        Deterministic guardrails to reduce POV drift and missing time markers.
        """
        if not text:
            return text

        out = text
        style = self._effective_style(style)
        protagonist = "수민" if "sumin" in protagonist_name.lower() else protagonist_name

        if style == "third_person_close":
            # Remove first-person POV leakage.
            replacements = [
                (r'(?<![가-힣])나는(?![가-힣])', f'{protagonist}은'),
                (r'(?<![가-힣])내가(?![가-힣])', f'{protagonist}이'),
                (r'(?<![가-힣])저는(?![가-힣])', f'{protagonist}은'),
                (r'(?<![가-힣])제가(?![가-힣])', f'{protagonist}이'),
                (r'(?<![가-힣])내(?![가-힣])', f'{protagonist}의'),
                (r'그녀', '그'),
            ]
            for pat, rep in replacements:
                out = re.sub(pat, rep, out)

        # Ensure at least minimal explicit time-flow markers.
        # Avoid hard-coded scene/location insertion that can create repetitive artifacts.
        time_marker = re.search(r'잠시 후|그 후|이후|그사이|한편|이윽고|곧이어|다음 날|그날 밤|며칠 후', out)
        if not time_marker:
            paragraphs = [p for p in out.split("\n\n") if p.strip()]
            if len(paragraphs) >= 4:
                bridge_options = [
                    "수민은 고개를 들었고, 장면의 공기는 다음 움직임으로 옮겨갔다.",
                    "의자가 살짝 밀리자 시선도 자연스럽게 다음 자리로 모였다.",
                    "문 쪽으로 고개가 돌아가면서 대화의 축도 다른 지점으로 옮겨갔다.",
                ]
                pick = sum(ord(ch) for ch in paragraphs[1]) % len(bridge_options)
                paragraphs.insert(2, bridge_options[pick])
                out = "\n\n".join(paragraphs)

        # Ensure date anchor appears for timeline coherence scoring.
        cfg_date = str(self.episode_config.get("date", "")).strip()
        if cfg_date:
            m = re.match(r"(\d{4})-(\d{2})-(\d{2})", cfg_date)
            if m:
                y, mo, da = m.group(1), str(int(m.group(2))), str(int(m.group(3)))
                date_phrase = f"{y}년 {mo}월 {da}일"
                if date_phrase not in out and y not in out:
                    out = f"{date_phrase}, 수민은 그날의 공기가 바뀌는 순간을 또렷하게 감지했다.\n\n{out}"

        return out

    def _cleanup_pov_reference_artifacts(
        self,
        text: str,
        style: str,
        protagonist_name: str,
    ) -> str:
        if not text:
            return text

        style = self._effective_style(style)
        if style != "first_person" and not self._feedback_needs_draft_cleanup():
            return text

        protagonist_short = "수민" if "sumin" in protagonist_name.lower() else protagonist_name.strip()
        alias_candidates = [
            protagonist_short,
            protagonist_name.strip(),
        ]
        alias_candidates = [alias for alias in alias_candidates if alias]
        unique_aliases: list[str] = []
        seen_aliases: set[str] = set()
        for alias in alias_candidates:
            key = alias.lower()
            if key in seen_aliases:
                continue
            seen_aliases.add(key)
            unique_aliases.append(alias)
        alias_pattern = "|".join(
            re.escape(alias)
            for alias in unique_aliases
        )
        if not alias_pattern:
            alias_pattern = re.escape(protagonist_short or protagonist_name or "수민")

        out_blocks: list[str] = []
        for block in [b.strip() for b in text.split("\n\n") if b.strip()]:
            sentences = self._split_korean_sentences(block)
            if not sentences:
                out_blocks.append(block)
                continue
            rebuilt: list[str] = []
            for sent in sentences:
                current = sent.strip()
                if not current:
                    continue
                if not re.match(r"^[\"“”'‘’]", current):
                    if style == "first_person":
                        current = re.sub(rf"^({alias_pattern})(?:은|는)(?=[\s,.;!?…]|$)", "나는", current)
                        current = re.sub(rf"^({alias_pattern})(?:이|가)(?=[\s,.;!?…]|$)", "내가", current)
                        current = re.sub(rf"^({alias_pattern})의(?=[\s,.;!?…]|$)", "내", current)
                        current = re.sub(r"^(그는|그녀는)(?=[\s,.;!?…]|$)", "나는", current)
                        current = re.sub(r"^(그가|그녀가)(?=[\s,.;!?…]|$)", "내가", current)
                        current = re.sub(r"^(그의|그녀의)(?=[\s,.;!?…]|$)", "내", current)
                    if self._feedback_needs_draft_cleanup():
                        current = re.sub(r"\b수민는\b", "수민은", current)
                        current = re.sub(r"\b단어은\b", "단어는", current)
                rebuilt.append(current)
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _reduce_local_repetition(self, text: str) -> str:
        """
        Deterministic cleanup for local repetition that often survives LLM polish.
        Removes near-duplicate adjacent sentences while preserving event order.
        """
        if not text:
            return text

        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        recent_sentence_fp: list[str] = []
        fp_window = 5 if self._feedback_mentions(
            "반복",
            "중복",
            "반복되는 표현",
            "비슷한 상황",
            "비슷한 상황과 묘사",
            "묘사가 반복",
            "지루",
        ) else 3

        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue

            block_fp = self._block_fingerprint(block)
            if block_fp and out_blocks:
                prev_block_fp = self._block_fingerprint(out_blocks[-1])
                if self._is_near_duplicate_sentence(block_fp, prev_block_fp):
                    continue

            sentences = self._split_korean_sentences(block)
            if not sentences:
                continue

            keep: list[str] = []
            for sent in sentences:
                fp = self._sentence_fingerprint(sent)
                if fp and any(self._is_near_duplicate_sentence(fp, prev) for prev in recent_sentence_fp):
                    continue
                keep.append(sent)
                if fp:
                    recent_sentence_fp.append(fp)
                    if len(recent_sentence_fp) > fp_window:
                        recent_sentence_fp.pop(0)

            if keep:
                out_blocks.append(self._compress_repeated_opening_phrases(" ".join(keep)))

        return "\n\n".join(out_blocks)

    def _compress_repeated_opening_phrases(self, text: str) -> str:
        """
        Remove adjacent prose sentences that reopen the same beat without adding
        a fresh action, decision, or consequence.
        """
        if not text:
            return text

        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue

            sentences = self._split_korean_sentences(block)
            if len(sentences) < 2:
                out_blocks.append(block)
                continue

            rebuilt: list[str] = []
            seen_openers: list[str] = []
            for sent in sentences:
                current = sent.strip()
                if not current:
                    continue
                opener = self._sentence_opening_signature(current)
                repeated_opener = bool(opener and opener in seen_openers[-3:])
                if repeated_opener and not re.search(r"[\"“”'‘’]", current):
                    if self._sentence_has_action_or_decision(current):
                        if rebuilt and not self._sentence_has_action_or_decision(rebuilt[-1]):
                            rebuilt[-1] = current
                    elif self._is_explanation_like_sentence(current) and rebuilt:
                        rebuilt[-1] = self._prefer_stronger_tension_sentence(rebuilt[-1], current)
                    continue
                rebuilt.append(current)
                if opener:
                    seen_openers.append(opener)
                    if len(seen_openers) > 4:
                        seen_openers.pop(0)

            if rebuilt:
                out_blocks.append(" ".join(rebuilt).strip())

        return "\n\n".join(b for b in out_blocks if b.strip())

    def _compress_repeated_tension_beats(self, text: str) -> str:
        """
        Collapse adjacent sentences that restate the same tension without
        adding a new action, decision, or consequence.
        """
        if not text:
            return text

        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue
            sentences = self._split_korean_sentences(block)
            if len(sentences) < 2:
                out_blocks.append(block)
                continue

            rebuilt: list[str] = []
            seen_fp: set[str] = set()
            for sent in sentences:
                current = sent.strip()
                if not current:
                    continue
                fp = self._sentence_fingerprint(current)
                if fp and fp in seen_fp and not self._sentence_has_action_or_decision(current):
                    continue
                current_emotion = self._emotion_signature(current)
                last_emotion = self._emotion_signature(rebuilt[-1]) if rebuilt else ""
                if (
                    rebuilt
                    and current_emotion
                    and current_emotion == last_emotion
                    and not self._sentence_has_action_or_decision(current)
                    and not self._sentence_has_action_or_decision(rebuilt[-1])
                ):
                    rebuilt[-1] = self._prefer_stronger_tension_sentence(rebuilt[-1], current)
                    if fp:
                        seen_fp.add(fp)
                    continue
                if rebuilt and self._is_redundant_tension_restatement(rebuilt[-1], current):
                    rebuilt[-1] = self._prefer_stronger_tension_sentence(rebuilt[-1], current)
                else:
                    rebuilt.append(current)
                if fp:
                    seen_fp.add(fp)
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _trim_post_metaphor_explanations(self, text: str) -> str:
        if not text:
            return text
        trim_metaphor_explanations = (
            self._feedback_flag_enabled("avoid_metaphor_explanation")
            or self._feedback_mentions("비유", "은유", "의미를 다시 설명", "문단 밀도", "호흡이 무거워")
        )
        strip_meta_markers = (
            self._feedback_flag_enabled("strip_meta_markers")
            or self._feedback_mentions("메타 표식", "작업 메모", "ep01의 온도계", "ep01—scene21", "완성 원고")
        )
        if not (trim_metaphor_explanations or strip_meta_markers):
            return text

        out_blocks: list[str] = []
        for block in [b.strip() for b in text.split("\n\n") if b.strip()]:
            cleaned_block = self._strip_meta_marker_artifacts(block) if strip_meta_markers else block
            if not cleaned_block:
                continue
            sentences = self._split_korean_sentences(cleaned_block)
            if strip_meta_markers:
                sentences = [s for s in sentences if not self._is_meta_marker_sentence(s)]
                if not sentences:
                    continue
            if len(sentences) < 2:
                if trim_metaphor_explanations and sentences:
                    single = sentences[0].strip()
                    if (
                        self._is_explanation_like_sentence(single)
                        and not self._sentence_has_action_or_decision(single)
                    ):
                        shortened_single = self._shorten_explanatory_sentence(single)
                        if shortened_single:
                            out_blocks.append(shortened_single)
                            continue
                out_blocks.append(cleaned_block)
                continue
            kept: list[str] = []
            explanation_taken = False
            idx = 0
            while idx < len(sentences):
                current = sentences[idx].strip()
                if not current:
                    idx += 1
                    continue
                if (
                    trim_metaphor_explanations
                    and idx > 0
                    and self._is_metaphor_like_sentence(sentences[idx - 1])
                    and self._is_explanation_like_sentence(current)
                    and not self._sentence_has_action_or_decision(current)
                ):
                    shortened = self._shorten_explanatory_sentence(current)
                    if shortened and not explanation_taken:
                        kept.append(shortened)
                        explanation_taken = True
                    idx += 1
                    continue
                if (
                    trim_metaphor_explanations
                    and self._is_explanation_like_sentence(current)
                    and not self._is_metaphor_like_sentence(current)
                    and not self._sentence_has_action_or_decision(current)
                    and kept
                    and (
                        self._is_metaphor_like_sentence(kept[-1])
                        or self._is_explanation_like_sentence(kept[-1])
                    )
                    and not (idx > 0 and self._is_metaphor_like_sentence(sentences[idx - 1]))
                ):
                    shortened = self._shorten_explanatory_sentence(current)
                    if not explanation_taken and shortened:
                        kept.append(shortened)
                        explanation_taken = True
                    idx += 1
                    continue
                if (
                    trim_metaphor_explanations
                    and self._is_explanation_like_sentence(current)
                    and not self._is_metaphor_like_sentence(current)
                    and not self._sentence_has_action_or_decision(current)
                    and not (idx > 0 and self._is_metaphor_like_sentence(sentences[idx - 1]))
                ):
                    shortened = self._shorten_explanatory_sentence(current)
                    if shortened and not explanation_taken:
                        kept.append(shortened)
                        explanation_taken = True
                    elif shortened and self._is_static_inner_monologue_sentence(shortened) and kept:
                        kept.append(shortened)
                    idx += 1
                    continue
                if trim_metaphor_explanations and idx + 1 < len(sentences) and self._is_post_metaphor_explanation_pair(current, sentences[idx + 1].strip()):
                    shortened = self._shorten_explanatory_sentence(current)
                    kept.append(shortened or current)
                    explanation_taken = True
                    idx += 1
                    while idx < len(sentences):
                        nxt = sentences[idx].strip()
                        if (
                            self._is_explanation_like_sentence(nxt)
                            and not self._sentence_has_action_or_decision(nxt)
                        ):
                            if not explanation_taken and self._sentence_has_concrete_anchor(nxt):
                                kept.append(nxt)
                                explanation_taken = True
                            idx += 1
                            continue
                        if (
                            self._is_pressure_heavy_sentence(nxt)
                            and not self._sentence_has_action_or_decision(nxt)
                        ):
                            next_short = self._shorten_explanatory_sentence(nxt)
                            if not explanation_taken and next_short:
                                kept.append(next_short)
                                explanation_taken = True
                            idx += 1
                            continue
                        break
                    continue
                kept.append(current)
                if self._is_explanation_like_sentence(current):
                    explanation_taken = True
                idx += 1
            out_blocks.append(" ".join(s for s in kept if s).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    @staticmethod
    def _meta_marker_patterns() -> tuple[re.Pattern[str], ...]:
        return (
            re.compile(r"(?i)\bep\s*\d{1,2}\s*[—\-_:]\s*scene\s*\d+\b"),
            re.compile(r"(?i)\bscene\s*[-_ ]?\d+\b"),
            re.compile(r"(?i)\bturn\s*[-_ ]?\d+\b"),
            re.compile(r"(?i)\bphase\s*[-_ ]?\d+\b"),
            re.compile(r"(?i)\b(?:ep|episode)\s*\d{1,2}\s*의\s*온도계\b"),
            re.compile(r"(?i)\b(?:ep|episode)\s*\d{1,2}\b"),
            re.compile(r"(?i)\b(?:metadata|meta tag|draft note|work note)\b"),
            re.compile(r"(?i)\b(?:scene|episode)\s*note\b"),
        )

    def _strip_meta_marker_artifacts(self, text: str) -> str:
        cleaned_sentences: list[str] = []
        for raw in self._split_korean_sentences(text):
            current = str(raw or "").strip()
            if not current:
                continue
            stripped = current
            for pattern in self._meta_marker_patterns():
                stripped = pattern.sub(" ", stripped)
            stripped = re.sub(r"[\[(]\s*(?:ep|scene|turn|phase)[^)\]]*[\])]", " ", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s{2,}", " ", stripped).strip()
            content_only = stripped.strip(" ,;:.-")
            if not re.search(r"[0-9A-Za-z가-힣]", content_only):
                continue
            if not content_only or self._is_meta_marker_sentence(content_only):
                continue
            cleaned_sentences.append(content_only + stripped[len(content_only):] if stripped.startswith(content_only) else content_only)
        return " ".join(cleaned_sentences).strip()

    def _is_meta_marker_sentence(self, sentence: str) -> bool:
        low = re.sub(r"\s+", " ", str(sentence or "").strip().lower())
        if not low:
            return False
        token_count = len(re.findall(r"[0-9a-z가-힣]+", low))
        meta_hits = sum(1 for pattern in self._meta_marker_patterns() if pattern.search(low))
        if meta_hits >= 1 and token_count <= 8:
            return True
        if re.match(r"^(?:ep|scene|turn|phase)\s*[-_:]?\s*\d+", low):
            return True
        return bool(re.match(r"^(?:작업 메모|메타 표식|draft note|work note|scene note|episode note)\b", low))

    def _has_post_metaphor_explanation_pairs(self, text: str) -> bool:
        sentences = self._split_korean_sentences(text)
        return any(
            self._is_post_metaphor_explanation_pair(sentences[idx], sentences[idx + 1])
            for idx in range(len(sentences) - 1)
        )

    def _is_post_metaphor_explanation_pair(self, first: str, second: str) -> bool:
        first_clean = str(first or "").strip()
        second_clean = str(second or "").strip()
        if not first_clean or not second_clean:
            return False
        if re.search(r"[\"“”'‘’]", first_clean + second_clean):
            return False
        metaphor_like = self._is_metaphor_like_sentence(first_clean)
        explanation_like = self._is_explanation_like_sentence(second_clean)
        if not metaphor_like or not explanation_like:
            return False
        if self._sentence_has_action_or_decision(second_clean):
            return False
        return True

    @staticmethod
    def _is_metaphor_like_sentence(sentence: str) -> bool:
        return bool(re.search(r"(마치|흡사|처럼|같았다|같은|듯했다|듯한)", str(sentence or "").strip()))

    @staticmethod
    def _sentence_has_concrete_anchor(sentence: str) -> bool:
        return bool(re.search(
            r"(배지|명함|문서|조항|계약|규정|허가|출입|번호|수치|모니터|경보|문|복도|방|회의실|발표장|시선|경비|보안|기관|부서|관리자|관찰자|감시 장치)",
            str(sentence or "").lower(),
        ))

    @staticmethod
    def _is_explanation_like_sentence(sentence: str) -> bool:
        return bool(re.search(
            r"^(즉|다시 말해|쉽게 말하면|그 말은|그 뜻은|결국|한마디로)|"
            r"(라는 뜻|뜻이었다|의미였|셈이었다|말이었다|다름 아니었다|뜻에 가까웠다)",
            str(sentence or "").strip(),
        ))

    def _shorten_explanatory_sentence(self, sentence: str) -> str:
        """
        Reduce an explanation sentence to the shortest usable factual clause.
        Returns an empty string when the sentence is only restating meaning.
        """
        text = re.sub(r"\s+", " ", str(sentence or "").strip())
        if not text:
            return ""
        text = re.sub(r"^(?:즉|다시 말해|쉽게 말하면|그 말은|그 뜻은|결국|한마디로)\s*", "", text)
        text = re.split(
            r"(?:라는 뜻|뜻이었다|의미였|셈이었다|말이었다|다름 아니었다|뜻에 가까웠다)",
            text,
            maxsplit=1,
        )[0].strip()
        text = re.split(r"[,;]|(?:\s+그리고\s+)|(?:\s+하지만\s+)|(?:\s+그러나\s+)", text, maxsplit=1)[0].strip()
        if not text:
            return ""
        if not self._sentence_has_action_or_decision(text) and not self._sentence_has_concrete_anchor(text):
            return ""
        return self._ensure_summary_sentence(text)

    def _is_redundant_tension_restatement(self, left: str, right: str) -> bool:
        left_clean = str(left or "").strip()
        right_clean = str(right or "").strip()
        if not left_clean or not right_clean:
            return False
        if re.search(r"[\"“”'‘’]", left_clean + right_clean):
            return False

        left_tokens = self._sentence_token_set(left_clean)
        right_tokens = self._sentence_token_set(right_clean)
        if not left_tokens or not right_tokens:
            return False

        overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
        same_emotion = bool(self._emotion_signature(left_clean)) and (
            self._emotion_signature(left_clean) == self._emotion_signature(right_clean)
        )
        same_pressure = self._is_pressure_heavy_sentence(left_clean) and self._is_pressure_heavy_sentence(right_clean)
        both_static = not self._sentence_has_action_or_decision(left_clean) and not self._sentence_has_action_or_decision(right_clean)
        return (same_emotion or same_pressure) and both_static and overlap >= 0.4

    def _prefer_stronger_tension_sentence(self, left: str, right: str) -> str:
        left_has_action = self._sentence_has_action_or_decision(left)
        right_has_action = self._sentence_has_action_or_decision(right)
        if left_has_action != right_has_action:
            return right if right_has_action else left

        left_score = self._pressure_sentence_score(left)
        right_score = self._pressure_sentence_score(right)
        if right_score != left_score:
            return right if right_score > left_score else left

        left_words = self._sentence_word_count(left)
        right_words = self._sentence_word_count(right)
        if right_words != left_words:
            return right if right_words < left_words else left
        return left

    @staticmethod
    def _sentence_token_set(sentence: str) -> set[str]:
        cleaned = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", str(sentence or "").lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        stop = {
            "그리고", "그러자", "다만", "하지만", "그러나", "한편", "이어서", "그", "그의",
            "수민은", "수민이", "그는", "그가", "그녀는", "그녀가",
        }
        return {
            token for token in cleaned.split()
            if len(token) >= 2 and token not in stop
        }

    @staticmethod
    def _is_pressure_heavy_sentence(sentence: str) -> bool:
        return bool(re.search(
            r"(긴장|압박|불안|초조|정적|침묵|차갑|날카롭|굳었|멈칫|숨을 죽|턱선|경계|서늘)",
            str(sentence or "").lower(),
        ))

    def _pressure_sentence_score(self, sentence: str) -> int:
        score = 0
        if self._sentence_has_action_or_decision(sentence):
            score += 3
        if self._is_pressure_heavy_sentence(sentence):
            score += 2
        if re.search(r"(드러났|밝혀졌|확인됐|선택|결정|질문|대답|응답|거절|수락)", str(sentence or "")):
            score += 2
        return score

    def _trim_redundant_sensory_sentences(self, text: str) -> str:
        if not text:
            return text

        channel_cap = self._feedback_sensory_channel_cap(default=2)
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue
            sentences = self._split_korean_sentences(block)
            if len(sentences) < 2:
                out_blocks.append(block)
                continue

            kept: list[str] = []
            channel_counts: dict[str, int] = {}
            prev_fp = ""
            for sent in sentences:
                channel = self._dominant_sensory_channel(sent)
                fp = self._sentence_fingerprint(sent)
                if (
                    channel
                    and self._is_sensory_heavy_sentence(sent)
                    and channel_counts.get(channel, 0) >= channel_cap
                    and (fp == prev_fp or not self._sentence_has_action_or_decision(sent))
                ):
                    continue
                kept.append(sent)
                if channel and self._is_sensory_heavy_sentence(sent):
                    channel_counts[channel] = channel_counts.get(channel, 0) + 1
                if fp:
                    prev_fp = fp
            out_blocks.append(" ".join(kept).strip() if kept else block)
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _trim_redundant_emotion_sentences(self, text: str) -> str:
        if not text:
            return text

        repeat_cap = self._feedback_emotion_repeat_cap(default=1)
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue
            sentences = self._split_korean_sentences(block)
            if len(sentences) < 2:
                out_blocks.append(block)
                continue

            kept: list[str] = []
            emotion_counts: dict[str, int] = {}
            prev_fp = ""
            for sent in sentences:
                sig = self._emotion_signature(sent)
                fp = self._sentence_fingerprint(sent)
                if (
                    sig
                    and emotion_counts.get(sig, 0) >= repeat_cap
                    and (fp == prev_fp or not self._sentence_has_action_or_decision(sent))
                ):
                    continue
                kept.append(sent)
                if sig:
                    emotion_counts[sig] = emotion_counts.get(sig, 0) + 1
                if fp:
                    prev_fp = fp
            out_blocks.append(" ".join(kept).strip() if kept else block)
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _is_sensory_heavy_sentence(self, sentence: str) -> bool:
        low = str(sentence or "").lower()
        return bool(self._dominant_sensory_channel(sentence)) and bool(
            re.search(r"(빛|시선|공기|차갑|냉기|열기|손끝|숨|침묵|기계음|울림|그림자|소리)", low)
        )

    @staticmethod
    def _sentence_has_action_or_decision(sentence: str) -> bool:
        return bool(re.search(
            r"(건넸|밀었|열었|접었|돌렸|움직였|받았|붙잡|꺼냈|멈추|확인|드러났|알아차렸|결정|선택|반응|질문|대답|응답|설득|거절|고개를 들었|숨을 골랐)",
            str(sentence or ""),
        ))

    @staticmethod
    def _emotion_signature(sentence: str) -> str:
        low = str(sentence or "").lower()
        groups = {
            "tension": r"(긴장|불안|초조|압박|조여|굳었|버텼|날카로웠)",
            "confusion": r"(망설|당황|혼란|멈칫|주저|흔들렸)",
            "relief": r"(안도|숨을 골랐|어깨 힘|진정|누그러졌)",
            "anger": r"(분노|짜증|쏘아붙|날을 세웠|턱선이 굳)",
            "resolve": r"(결정|선택|다짐|버텨|받아치)",
        }
        for name, pattern in groups.items():
            if re.search(pattern, low):
                return name
        return ""

    def _diversify_transition_openers(self, text: str) -> str:
        if not text:
            return text
        cap = self._feedback_transition_opener_cap(default=2)
        avoid_terms = self._feedback_transition_avoid_terms()
        out_blocks: list[str] = []
        for block in [b.strip() for b in text.split("\n\n") if b.strip()]:
            sentences = self._split_korean_sentences(block)
            if not sentences:
                out_blocks.append(block)
                continue
            rebuilt: list[str] = []
            recent_connectors: list[str] = []
            connector_counts: dict[str, int] = {}
            for sent in sentences:
                connector = self._sentence_leading_connector(sent)
                connector_key = connector.lower()
                needs_swap = bool(connector) and (
                    connector_key in avoid_terms
                    or connector_counts.get(connector_key, 0) >= cap
                    or (recent_connectors and recent_connectors[-1] == connector_key)
                )
                if needs_swap:
                    replacement = self._pick_transition_replacement(
                        connector,
                        recent_connectors,
                        connector_counts,
                    )
                    pattern = re.escape(connector)
                    if replacement:
                        sent = re.sub(
                            rf"^([\"“”'‘’\(\)\[\]\s]*){pattern}\s*,?\s*",
                            rf"\1{replacement} ",
                            sent.strip(),
                            count=1,
                        ).strip()
                        connector = replacement
                        connector_key = connector.lower()
                    else:
                        sent = re.sub(
                            rf"^([\"“”'‘’\(\)\[\]\s]*){pattern}\s*,?\s*",
                            r"\1",
                            sent.strip(),
                            count=1,
                        ).strip()
                        connector = ""
                        connector_key = ""
                rebuilt.append(sent)
                if connector_key:
                    connector_counts[connector_key] = connector_counts.get(connector_key, 0) + 1
                    recent_connectors.append(connector_key)
                    if len(recent_connectors) > 2:
                        recent_connectors.pop(0)
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())
        return "\n\n".join(out_blocks)

    @staticmethod
    def _transition_replacement_catalog() -> dict[str, list[str]]:
        return {
            "그리고": ["말끝이 가라앉자", "문 쪽에서 인기척이 일자", "의자 다리가 짧게 끌리자", "정면을 버틴 채", "짧은 숨이 스친 뒤", "답이 바로 나오지 않는 사이"],
            "그러자": ["말끝이 떨어지자", "문 쪽에서 인기척이 일자", "의자 다리가 짧게 끌리자", "짧은 정적 끝에", "정면을 버틴 채", "답이 바로 나오지 않는 사이"],
            "다만": ["대신", "문제는", "그래도", "한편"],
            "이어서": ["누군가 펜을 내려놓자", "말끝이 가라앉자", "의자 다리가 짧게 끌리자", "정면을 버틴 채", "짧은 숨이 스친 뒤", "답이 바로 나오지 않는 사이"],
            "그 순간": ["문 쪽에서 인기척이 일자", "말끝이 떨어지자", "누군가 펜을 내려놓자", "짧은 정적 끝에", "의자 다리가 짧게 끌리자", "실내의 공기가 식자"],
        }

    def _pick_transition_replacement(
        self,
        connector: str,
        recent_connectors: list[str],
        connector_counts: dict[str, int],
    ) -> str:
        options = self._transition_replacement_catalog().get(str(connector or "").strip(), [])
        if not options:
            return ""
        avoid_terms = self._feedback_transition_avoid_terms()
        cap = self._feedback_transition_opener_cap(default=2)
        recent_last = recent_connectors[-1] if recent_connectors else ""
        for candidate in options:
            key = candidate.lower()
            if key in avoid_terms:
                continue
            if key == recent_last:
                continue
            if connector_counts.get(key, 0) >= cap:
                continue
            return candidate
        for candidate in options:
            key = candidate.lower()
            if key in avoid_terms:
                continue
            if key == recent_last:
                continue
            return candidate
        return ""

    def _merge_clipped_sentence_runs(self, text: str) -> str:
        """
        Merge repeated short narrative fragments into one flowing sentence.
        This also collapses duplicate inner-pressure beats so one strong line carries
        the feeling before the prose pivots into action or consequence.
        """
        if not text:
            return text

        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue
            sentences = self._split_korean_sentences(block)
            if len(sentences) < 2:
                out_blocks.append(block)
                continue

            rebuilt: list[str] = []
            run: list[str] = []
            for sent in sentences:
                if self._is_mergeable_clipped_sentence(sent):
                    run.append(sent.strip())
                    continue
                pruned_run = self._prune_clipped_sentence_run(run)
                pruned_run = self._prefer_concrete_clipped_run(pruned_run)
                if len(pruned_run) >= 2:
                    rebuilt.append(self._combine_clipped_sentence_run(pruned_run))
                else:
                    rebuilt.extend(pruned_run)
                run = []
                rebuilt.append(sent.strip())
            pruned_run = self._prune_clipped_sentence_run(run)
            pruned_run = self._prefer_concrete_clipped_run(pruned_run)
            if len(pruned_run) >= 2:
                rebuilt.append(self._combine_clipped_sentence_run(pruned_run))
            else:
                rebuilt.extend(pruned_run)
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())

        return "\n\n".join(b for b in out_blocks if b.strip())

    def _prefer_concrete_clipped_run(self, sentences: list[str]) -> list[str]:
        if len(sentences) < 2:
            return sentences
        if not any(self._is_jargon_heavy_sentence(sent) for sent in sentences):
            return sentences

        def score(sentence: str) -> tuple[int, int]:
            value = 0
            if self._sentence_has_action_or_decision(sentence):
                value += 4
            if self._is_pressure_heavy_sentence(sentence):
                value += 2
            if self._sentence_image_signature(sentence):
                value += 2
            if not self._is_jargon_heavy_sentence(sentence):
                value += 1
            return value, -self._sentence_word_count(sentence)

        anchor = max(sentences, key=score)
        follow = ""
        for sent in sentences:
            if sent == anchor:
                continue
            if self._sentence_has_action_or_decision(sent) or self._sentence_image_signature(sent):
                follow = sent
                break
        if not follow:
            follow = next((sent for sent in sentences if sent != anchor), "")
        reduced = [anchor]
        if follow and follow != anchor:
            reduced.append(follow)
        return reduced

    def _is_mergeable_clipped_sentence(self, sentence: str) -> bool:
        sent = str(sentence or "").strip()
        if not sent:
            return False
        if re.search(r"[\"“”'‘’]", sent):
            return False
        if len(re.findall(r"[0-9A-Za-z가-힣]+", sent)) > 6:
            return False
        if re.search(r"(?:^|[\s])(?:그리고|하지만|그러나|다만|한편|또는)\b", sent):
            return False
        return True

    def _prune_clipped_sentence_run(self, sentences: list[str]) -> list[str]:
        kept: list[str] = []
        seen_fp: set[str] = set()
        inner_idx: Optional[int] = None
        last_image_sig = ""
        keep_single_inner = self._feedback_flag_enabled("single_strong_interior_beat") or self._feedback_mentions(
            "내면 독백",
            "한 번만 강하게",
            "재진술",
            "같은 정보와 감정",
        )
        for raw in sentences:
            current = str(raw or "").strip()
            if not current:
                continue
            fp = self._sentence_fingerprint(current)
            if fp and fp in seen_fp:
                continue
            if kept and self._is_redundant_tension_restatement(kept[-1], current):
                kept[-1] = self._prefer_stronger_tension_sentence(kept[-1], current)
                last_image_sig = self._sentence_image_signature(kept[-1])
                if fp:
                    seen_fp.add(fp)
                continue
            current_image_sig = self._sentence_image_signature(current)
            if (
                kept
                and current_image_sig
                and last_image_sig
                and not self._sentence_has_action_or_decision(current)
                and self._image_signature_overlap(last_image_sig, current_image_sig)
            ):
                kept[-1] = self._prefer_stronger_tension_sentence(kept[-1], current)
                last_image_sig = self._sentence_image_signature(kept[-1])
                if fp:
                    seen_fp.add(fp)
                continue
            is_inner = self._is_static_inner_monologue_sentence(current)
            if keep_single_inner and is_inner and inner_idx is not None:
                kept[inner_idx] = self._prefer_stronger_tension_sentence(kept[inner_idx], current)
                continue
            kept.append(current)
            if fp:
                seen_fp.add(fp)
            if is_inner and inner_idx is None:
                inner_idx = len(kept) - 1
            if current_image_sig:
                last_image_sig = current_image_sig
        if len(kept) >= 2:
            non_action = [sent for sent in kept if not self._sentence_has_action_or_decision(sent)]
            if len(non_action) >= 2 and all(
                self._is_static_inner_monologue_sentence(sent) or self._is_pressure_heavy_sentence(sent)
                for sent in non_action
            ):
                best = max(
                    kept,
                    key=lambda sent: (
                        self._pressure_sentence_score(sent),
                        -self._sentence_word_count(sent),
                    ),
                )
                return [best]
        return kept

    def _derive_clipped_bridge_fragment(self, sentences: list[str]) -> str:
        """
        Derive a short bridge fragment from a recent sentence when the source
        already contains a natural pause or causal turn.
        """
        for raw in reversed(sentences[-3:]):
            current = re.sub(r"\s+", " ", str(raw or "")).strip()
            if not current:
                continue
            if not re.search(r"[,:;—~]", current):
                continue
            fragment = re.split(r"[,:;—~]\s*", current, maxsplit=1)[0].strip()
            fragment = re.sub(r"[.!?…]+$", "", fragment).strip()
            if not fragment or fragment == current:
                continue
            if self._sentence_word_count(fragment) < 2:
                continue
            if self._sentence_has_action_or_decision(fragment) or self._is_pressure_heavy_sentence(fragment):
                return fragment
        return ""

    @staticmethod
    def _sentence_image_signature(sentence: str) -> str:
        low = re.sub(r"\s+", " ", str(sentence or "")).strip().lower()
        if not low:
            return ""
        groups = [
            ("space", ("복도", "문", "문턱", "계단", "방", "회의실", "발표장", "무대", "엘리베이터", "통로")),
            ("atmosphere", ("공기", "정적", "침묵", "소음", "냉기", "열기", "온기", "기류", "공조")),
            ("gaze", ("시선", "눈", "응시", "바라")),
            ("body", ("숨", "호흡", "손끝", "손바닥", "어깨", "턱", "입술", "등")),
            ("sound", ("소리", "기계음", "발소리", "웅", "삐", "울림", "마찰")),
        ]
        hits: list[str] = []
        for group, tokens in groups:
            if any(tok in low for tok in tokens):
                hits.append(group)
        return "|".join(hits[:3])

    @staticmethod
    def _image_signature_overlap(left: str, right: str) -> bool:
        if not left or not right:
            return False
        left_tokens = {token for token in str(left).split("|") if token}
        right_tokens = {token for token in str(right).split("|") if token}
        return bool(left_tokens & right_tokens)

    def _is_static_inner_monologue_sentence(self, sentence: str) -> bool:
        sent = str(sentence or "").strip()
        if not sent or re.search(r"[\"“”'‘’]", sent):
            return False
        if self._sentence_has_action_or_decision(sent):
            return False
        if self._emotion_signature(sent):
            return True
        return bool(re.search(
            r"(느낌|생각|계산|판단|확신|의심|두려|불안|초조|경계|망설|끌렸|끌리면서도|선별당)",
            sent,
        ))

    def _combine_clipped_sentence_run(self, sentences: list[str]) -> str:
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0]
        ordered: list[str] = []
        seen_openers: set[str] = set()
        for raw in sentences:
            current = str(raw or "").strip()
            if not current:
                continue
            opener = self._sentence_opening_signature(current)
            if opener and opener in seen_openers:
                continue
            if ordered and self._is_redundant_tension_restatement(ordered[-1], current):
                ordered[-1] = self._prefer_stronger_tension_sentence(ordered[-1], current)
                continue
            ordered.append(current)
            if opener:
                seen_openers.add(opener)
        if not ordered:
            return ""
        combined = re.sub(r"[.!?…]+$", "", ordered[0].strip())
        if not combined:
            return " ".join(s for s in ordered if s.strip())

        connector = ""
        if self._should_use_clipped_bridge(ordered):
            connector = self._select_clipped_bridge_phrase(ordered)

        for idx, sent in enumerate(ordered[1:], start=1):
            cleaned = re.sub(r"[.!?…]+$", "", sent.strip())
            if not cleaned:
                continue
            if idx == 1 and connector:
                combined = f"{combined}, {connector} {cleaned}"
            else:
                combined = f"{combined}, {cleaned}"
        return combined.strip(" ,") + "."

    def _should_use_clipped_bridge(self, sentences: list[str]) -> bool:
        if len(sentences) < 2:
            return False
        if any(self._sentence_has_action_or_decision(sent) for sent in sentences):
            return False
        if len(sentences) >= 3:
            return True
        return any(
            self._is_static_inner_monologue_sentence(sent) or self._is_pressure_heavy_sentence(sent)
            for sent in sentences
        )

    def _select_clipped_bridge_phrase(self, sentences: list[str]) -> str:
        fragment = self._derive_clipped_bridge_fragment(sentences)
        if fragment:
            return fragment
        options = [
            conn
            for conn in [
                "말끝이 가라앉자",
                "문 쪽에서 인기척이 일자",
                "의자 다리가 짧게 끌리자",
                "정면을 버틴 채",
                "짧은 숨이 스친 뒤",
                "답이 바로 나오지 않는 사이",
            ]
            if conn.lower() not in self._feedback_transition_avoid_terms()
        ]
        if not options:
            return ""

        combined = " ".join(str(sent or "") for sent in sentences).lower()
        tokens = set(re.findall(r"[0-9A-Za-z가-힣]+", combined))
        best = ""
        best_score = -10
        for idx, candidate in enumerate(options):
            low_candidate = candidate.lower()
            cand_tokens = set(re.findall(r"[0-9A-Za-z가-힣]+", low_candidate))
            score = len(tokens & cand_tokens)
            if "말끝" in low_candidate:
                score -= 1
            if any(tok in combined for tok in ("문이", "문 쪽", "문쪽", "door", "doorway", "발소리", "발걸음", "열렸", "열리")) and "문" in low_candidate:
                score += 3
            if any(tok in combined for tok in ("숨", "호흡", "막혔", "가라앉", "초조", "긴장")) and "숨" in low_candidate:
                score += 3
            if any(tok in combined for tok in ("의자", "자리", "앉", "밀리", "끌리")) and "의자" in low_candidate:
                score += 3
            if any(tok in combined for tok in ("질문", "대답", "조건", "응답", "반박", "수락", "거절")) and "답이 바로" in low_candidate:
                score += 2
            if any(tok in combined for tok in ("정적", "침묵", "멈칫", "멈췄", "버텼")) and "정면" in low_candidate:
                score += 1
            score += max(0, len(options) - idx - 1) * 0.01
            if score > best_score:
                best_score = score
                best = candidate
        return best

    @staticmethod
    def _sentence_length_band(sentence: str) -> str:
        wc = ProseGenerator._sentence_word_count(sentence)
        if wc <= 6:
            return "short"
        if wc <= 16:
            return "medium"
        return "long"

    def _stagger_sentence_rhythm(self, text: str) -> str:
        """
        Break up repeated short/long runs so scene cadence does not lock into
        the same beat length for too many sentences in a row.
        """
        if not text:
            return text

        window = self._feedback_sentence_variety_window(default=4)
        split_cap = max(14, self._feedback_sentence_word_cap(default=25) - 4)
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue
            sentences = self._split_korean_sentences(block)
            if len(sentences) < window:
                out_blocks.append(block)
                continue

            rebuilt: list[str] = []
            for sent in sentences:
                rebuilt.append(sent.strip())
                if len(rebuilt) < window:
                    continue
                recent = rebuilt[-window:]
                bands = [self._sentence_length_band(item) for item in recent]
                if len(set(bands)) != 1:
                    continue
                band = bands[-1]
                if band == "short":
                    left = rebuilt[-2]
                    right = rebuilt[-1]
                    if self._is_mergeable_clipped_sentence(left) and self._is_mergeable_clipped_sentence(right):
                        rebuilt = rebuilt[:-2] + [self._combine_clipped_sentence_run([left, right])]
                elif band == "long":
                    split = self._split_sentence_by_word_cap(rebuilt[-1], max_words=split_cap)
                    if len(split) > 1:
                        rebuilt = rebuilt[:-1] + split

            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _compress_redundant_jargon_sentences(self, text: str) -> str:
        """
        Compress adjacent jargon-heavy sentences when they restate the same concept.
        This reduces technical density before final paragraph normalization.
        """
        if not text:
            return text

        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue
            sentences = self._split_korean_sentences(block)
            if len(sentences) < 2:
                out_blocks.append(block)
                continue

            rebuilt: list[str] = []
            idx = 0
            while idx < len(sentences):
                current = sentences[idx].strip()
                if idx + 1 < len(sentences):
                    nxt = sentences[idx + 1].strip()
                    overlap = self._technical_overlap(current, nxt)
                    if overlap and self._is_jargon_heavy_sentence(current) and self._is_jargon_heavy_sentence(nxt):
                        merged = re.sub(r"[.!?…]+$", "", current)
                        follow = re.sub(r"^[\"“”'‘’\s]+|[.!?…]+$", "", nxt)
                        rebuilt.append(f"{merged}, 이어 {follow}.")
                        idx += 2
                        continue
                rebuilt.append(current)
                idx += 1
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    @staticmethod
    def _block_fingerprint(block: str) -> str:
        cleaned = str(block or "").lower()
        cleaned = re.sub(r"[^0-9a-z가-힣\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        toks = [t for t in cleaned.split() if len(t) > 1]
        return " ".join(toks[:36])

    @staticmethod
    def _sentence_fingerprint(sentence: str) -> str:
        cleaned = str(sentence or "").lower()
        cleaned = re.sub(r"[^0-9a-z가-힣\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        stop = {
            "그리고", "하지만", "그러나", "그래서", "정말", "아주", "매우", "조금",
            "the", "a", "an", "and", "or", "to", "is", "are",
        }
        toks = [t for t in cleaned.split() if len(t) > 1 and t not in stop]
        return " ".join(toks[:18])

    @staticmethod
    def _sentence_opening_signature(sentence: str, max_tokens: int = 6) -> str:
        cleaned = re.sub(r"^[\"“”'‘’\s\-\(\)\[\]]+", "", str(sentence or "").strip())
        cleaned = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        if not cleaned:
            return ""
        stop = {
            "그리고", "그러자", "다만", "한편", "이어서", "그", "그의", "그녀", "그는", "또",
            "하지만", "그러나", "그래서", "the", "a", "an", "and", "or",
        }
        toks = [t for t in cleaned.split() if len(t) > 1 and t not in stop]
        return " ".join(toks[:max_tokens])

    @staticmethod
    def _is_near_duplicate_sentence(a: str, b: str) -> bool:
        if not a or not b:
            return False
        ta = set(a.split())
        tb = set(b.split())
        if not ta or not tb:
            return False
        overlap = len(ta & tb) / max(len(ta | tb), 1)
        return overlap >= 0.8

    def _ensure_transition_marker(self, text: str) -> str:
        lo, hi = self._feedback_transition_char_window()
        fallback = "시선이 다음 움직임으로 옮겨갔다."
        if not text:
            return self._fit_char_window(fallback, lo, hi)
        out = text.strip()
        return self._fit_char_window(out, lo, hi)

    @staticmethod
    def _fit_char_window(text: str, min_chars: int, max_chars: int) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) > max_chars:
            cut = s[:max_chars]
            pivot = max(cut.rfind(" "), cut.rfind(","))
            if pivot >= max(3, min_chars - 1):
                cut = cut[:pivot].strip()
            cut = cut.rstrip(" ,")
            if not re.search(r"[.!?…]$", cut):
                cut += "."
            s = cut
        while len(s) < min_chars:
            s = (s.rstrip(".") + " 잠깐.") if len(s) < max_chars - 3 else (s + ".")
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) >= max_chars:
                break
        return s

    def _normalize_paragraphs(self, text: str) -> str:
        """
        Post-process overly long paragraphs so analyzer-facing structure stays stable.
        Keeps content intact and only adjusts paragraph breaks.
        """
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        normalized: list[str] = []

        max_sentences = self._readability_controls()["paragraph_max"]
        strict_breathing = self._feedback_mentions(
            "긴 문장", "문장이 길", "긴 문단", "문단이 길", "문단", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴"
        )
        max_chars = 220 if strict_breathing else 320

        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                normalized.append(block)
                continue

            sentences = self._split_korean_sentences(block)

            if len(sentences) <= max_sentences:
                if len(sentences) == 1 and len(sentences[0]) > max_chars:
                    sentences = self._split_long_sentence_soft(sentences[0], max_chars=max_chars)
                if len(sentences) <= max_sentences:
                    normalized.append(" ".join(sentences) if sentences else block)
                    continue

            # Split long blocks into configurable chunks to keep readable breathing points.
            chunk_buf: list[str] = []
            chunk_chars = 0
            for sent in sentences:
                s = sent.strip()
                if not s:
                    continue
                projected_chars = chunk_chars + len(s) + (1 if chunk_buf else 0)
                if chunk_buf and (len(chunk_buf) >= max_sentences or projected_chars > max_chars):
                    normalized.append(" ".join(chunk_buf).strip())
                    chunk_buf = []
                    chunk_chars = 0
                chunk_buf.append(s)
                chunk_chars += len(s) + 1
            if chunk_buf:
                normalized.append(" ".join(chunk_buf).strip())

        return "\n\n".join(normalized)

    def _enforce_sentence_word_caps(self, text: str, max_words: int = 25) -> str:
        """
        Split overlong prose sentences into 2-3 shorter beats by word count.
        This is a deterministic fallback when model output ignores breathing constraints.
        """
        if not text:
            return text

        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue
            sentences = self._split_korean_sentences(block)
            if not sentences:
                out_blocks.append(block)
                continue
            rebuilt: list[str] = []
            for sent in sentences:
                rebuilt.extend(self._split_sentence_by_word_cap(sent, max_words=max_words))
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    @staticmethod
    def _split_sentence_by_word_cap(sentence: str, max_words: int = 25) -> list[str]:
        s = str(sentence or "").strip()
        if not s:
            return []
        words = re.findall(r"[0-9A-Za-z가-힣]+", s)
        if len(words) <= max_words and any(ch in s for ch in ('"', "“", "”")):
            return [s]
        if len(words) <= max_words:
            comma_heavy = (s.count(",") + s.count("，") + s.count(";")) >= 2
            connective_heavy = len(re.findall(r"(그리고|그러나|하지만|다만|한편|또한|그래서|그러자)", s)) >= 2
            if not comma_heavy and not connective_heavy:
                return [s]

        # Prefer semantic boundaries first, then fallback to token slicing.
        clauses = re.split(
            r"(?<=[,，;])\s+|(?<=\))\s+|(?<=다)\s+(?=그|하지만|그리고|그러나|다만|한편|또한|대신|결국)",
            s,
        )
        clauses = [c.strip() for c in clauses if c.strip()]
        if len(clauses) <= 1:
            clauses = s.split()
            if len(clauses) <= max_words:
                return [s]
            out: list[str] = []
            for i in range(0, len(clauses), max_words):
                chunk = " ".join(clauses[i:i + max_words]).strip()
                if chunk:
                    out.append(chunk)
            return out or [s]

        out: list[str] = []
        buf: list[str] = []
        buf_words = 0
        for clause in clauses:
            c_words = len(re.findall(r"[0-9A-Za-z가-힣]+", clause))
            if buf and (buf_words + c_words > max_words):
                out.append(" ".join(buf).strip())
                buf = []
                buf_words = 0
            buf.append(clause)
            buf_words += c_words
        if buf:
            out.append(" ".join(buf).strip())
        return out or [s]

    def _split_dense_information_paragraphs(self, text: str) -> str:
        """
        Break dense info paragraphs into smaller chunks (roughly one key point per chunk).
        """
        if not text:
            return text
        chunk_sent_cap = self._feedback_dense_sentence_cap(default=2)
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                out_blocks.append(block)
                continue
            sentences = self._split_korean_sentences(block)
            if len(sentences) <= 2:
                out_blocks.append(block)
                continue
            dense_score = (
                len(re.findall(r"\b[A-Z]{2,8}\b", block))
                + len(re.findall(r"\d+(?:\.\d+)?", block))
                + block.count("(")
                + block.count(",")
            )
            if dense_score < 8:
                out_blocks.append(block)
                continue
            # Keep chunk size small for readability in dense paragraphs.
            chunk: list[str] = []
            for sent in sentences:
                chunk.append(sent.strip())
                if len(chunk) >= chunk_sent_cap:
                    out_blocks.append(" ".join(chunk).strip())
                    chunk = []
            if chunk:
                out_blocks.append(" ".join(chunk).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _cap_paragraph_term_repetition(self, text: str, max_per_paragraph: int = 2) -> str:
        """
        Cap repeated motif/jargon terms within a single paragraph.
        When over cap, drop the most redundant sentence-level repeats.
        """
        if not text:
            return text
        terms = self._feedback_repeat_terms() + self._feedback_jargon_terms()
        terms = [t for t in terms if t]
        if not terms:
            return text

        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for idx, block in enumerate(blocks, start=1):
            lowered = block.lower()
            over_terms = [
                t for t in terms
                if self._count_feedback_term_occurrences(lowered, t) > max_per_paragraph
            ]
            if not over_terms:
                out_blocks.append(block)
                continue
            logger.info(
                "Paragraph repetition cap triggered at #%d for terms: %s",
                idx,
                ", ".join(over_terms[:4]),
            )
            sentences = self._split_korean_sentences(block)
            if len(sentences) <= 1:
                out_blocks.append(block)
                continue
            kept: list[str] = []
            seen_counts: dict[str, int] = {t.lower(): 0 for t in over_terms}
            for sent in sentences:
                drop = False
                low_sent = sent.lower()
                for term in over_terms:
                    key = term.lower()
                    occ = self._count_feedback_term_occurrences(low_sent, term)
                    if occ <= 0:
                        continue
                    if seen_counts.get(key, 0) >= max_per_paragraph:
                        drop = True
                        break
                if drop:
                    continue
                kept.append(sent)
                for term in over_terms:
                    key = term.lower()
                    seen_counts[key] = seen_counts.get(key, 0) + self._count_feedback_term_occurrences(low_sent, term)
            out_blocks.append(" ".join(kept).strip() if kept else block)
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _enforce_jargon_onboarding_and_variation(self, text: str) -> str:
        """
        Keep recurring technical wording concise without forcing parenthetical glosses.
        """
        if not text:
            return text

        out = text
        for entry in self._term_variation_catalog():
            pattern = entry["pattern"]
            variants = entry["variants"]
            seen = 0

            def _repl(match: re.Match) -> str:
                nonlocal seen
                token = match.group(0)
                seen += 1
                if seen >= 2 and variants and seen % 2 == 0:
                    return variants[(seen - 2) % len(variants)]
                return token

            out = re.sub(pattern, _repl, out, flags=re.IGNORECASE)
        return self._trim_repeated_jargon_gloss_sentences(out)

    def _trim_repeated_jargon_gloss_sentences(self, text: str) -> str:
        if not text:
            return text

        out_blocks: list[str] = []
        explained_terms: set[str] = set()
        for block in [b.strip() for b in text.split("\n\n") if b.strip()]:
            sentences = self._split_korean_sentences(block)
            if len(sentences) < 2:
                out_blocks.append(block)
                continue

            kept: list[str] = []
            for sent in sentences:
                term_keys = self._jargon_sentence_keys(sent)
                explanation_like = self._sentence_is_jargon_explanation(sent)
                if (
                    explanation_like
                    and term_keys
                    and term_keys.issubset(explained_terms)
                    and not self._sentence_has_action_or_decision(sent)
                ):
                    continue
                kept.append(sent.strip())
                if explanation_like and term_keys:
                    explained_terms.update(term_keys)
            out_blocks.append(" ".join(s for s in kept if s.strip()).strip() or block)
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _jargon_sentence_keys(self, sentence: str) -> set[str]:
        low = str(sentence or "").lower()
        keys: set[str] = set()
        for entry in self._term_variation_catalog():
            variants = [
                str(variant or "").strip().lower()
                for variant in entry.get("variants", [])
                if str(variant or "").strip()
            ]
            if any(variant in low for variant in variants):
                keys.add(variants[0] if variants else str(entry.get("pattern", "")).lower())
        for term in self._feedback_jargon_terms():
            low_term = str(term or "").strip().lower()
            if low_term and low_term in low:
                keys.add(low_term)
        return keys

    @staticmethod
    def _sentence_is_jargon_explanation(sentence: str) -> bool:
        return bool(re.search(
            r"(즉|쉽게 말하면|쉽게 말해|다시 말해|뜻이었다|말이었다|셈이었다|의미는|라는 뜻|풀어 말하면)",
            str(sentence or ""),
        ))

    def _is_jargon_heavy_sentence(self, sentence: str) -> bool:
        return len(self._technical_term_set(sentence)) >= 2

    def _technical_overlap(self, left: str, right: str) -> set[str]:
        return self._technical_term_set(left) & self._technical_term_set(right)

    def _technical_term_set(self, text: str) -> set[str]:
        raw = str(text or "")
        if not raw.strip():
            return set()
        tokens = set(re.findall(r"\b[A-Z]{2,8}(?:-\d+)?\b", raw))
        low = raw.lower()
        for entry in self._term_variation_catalog():
            for variant in entry.get("variants", []):
                variant_text = str(variant or "").lower()
                if variant_text and variant_text in low:
                    tokens.add(variant_text)
        for term in self._feedback_jargon_terms():
            low_term = str(term or "").lower()
            if low_term and low_term in low:
                tokens.add(low_term)
        return tokens

    @staticmethod
    def _trim_gloss(gloss: str, max_chars: int = 20) -> str:
        g = re.sub(r"\s+", " ", str(gloss or "")).strip()
        if not g:
            return "짧은 뜻풀이"
        if len(g) <= max_chars:
            return g
        trimmed = g[:max_chars].rstrip(" ,")
        return trimmed or g[:max_chars]

    @staticmethod
    def _term_variation_catalog() -> list[dict[str, object]]:
        return [
            {
                "pattern": r"\bcoherence\b|코히런스|결맞음",
                "gloss": "양자상태 유지력",
                "variants": ["코히런스", "결맞음"],
            },
            {
                "pattern": r"\bdrift\b|드리프트|편차",
                "gloss": "시간 누적 편차",
                "variants": ["드리프트", "편차"],
            },
            {
                "pattern": r"\blatency\b|지연|응답 지연",
                "gloss": "반응까지 걸린 시간",
                "variants": ["지연", "응답 지연"],
            },
            {
                "pattern": r"\bqpu\b|양자 처리 칩",
                "gloss": "양자 계산 핵심 칩",
                "variants": ["QPU", "양자 처리 칩"],
            },
            {
                "pattern": r"\bt[2₂]\b|T₂|T2",
                "gloss": "양자 상태 유지 시간",
                "variants": ["T2", "T₂"],
            },
        ]

    def _insert_short_beats_after_long_streak(
        self,
        text: str,
        long_threshold: int = 22,
        streak_limit: int = 2,
    ) -> str:
        if not text:
            return text
        min_short, max_short = self._feedback_short_beat_char_window()
        _, max_per_scene = self._feedback_short_beats_per_scene()
        if self._feedback_mentions(
            "짧게 끊기는 문장",
            "문장이 너무 자주 끊기",
            "짧은 반복 문장",
            "비슷한 리듬",
            "같은 리듬",
            "단조로운 리듬",
        ):
            long_threshold = max(long_threshold, 26)
            streak_limit = max(streak_limit, 3)
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        beat_idx = 0
        for block in blocks:
            sentences = self._split_korean_sentences(block)
            if not sentences:
                out_blocks.append(block)
                continue
            rebuilt: list[str] = []
            streak = 0
            inserted = 0
            for sent in sentences:
                rebuilt.append(sent)
                wc = self._sentence_word_count(sent)
                if wc >= long_threshold:
                    streak += 1
                else:
                    streak = 0
                if streak > streak_limit and inserted < max_per_scene:
                    tone = self._bridge_tone_for_context(rebuilt[-2:])
                    rebuilt.append(
                        self._rhythm_bridge_sentence(
                            beat_idx,
                            min_short,
                            max_short,
                            tone=tone,
                            context_sentences=rebuilt[-3:],
                        )
                    )
                    beat_idx += 1
                    inserted += 1
                    streak = 0
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    @staticmethod
    def _sentence_intensity_score(sentence: str) -> int:
        return len(re.findall(
            r"(긴장|압박|정적|침묵|날카|버텼|몰아붙|흔들|초조|불안|결정|선택|반박|거절|수락)",
            str(sentence or ""),
        ))

    def _bridge_tone_for_context(self, recent_sentences: list[str]) -> str:
        if not recent_sentences:
            return "strong"
        intensity = sum(self._sentence_intensity_score(sent) for sent in recent_sentences[-2:])
        return "plain" if intensity >= 2 else "strong"

    def _rhythm_bridge_sentence(
        self,
        idx: int,
        min_chars: int = 5,
        max_chars: int = 10,
        tone: str = "strong",
        context_sentences: Optional[list[str]] = None,
    ) -> str:
        min_chars = max(min_chars, 14)
        max_chars = max(max_chars, 28)
        contextual = self._derive_clipped_bridge_fragment(context_sentences or [])
        if contextual:
            bridge = contextual if re.search(r"[.!?…]$", contextual) else f"{contextual}."
            return self._fit_char_window(bridge, min_chars, max_chars)
        if context_sentences:
            recent = [str(sent or "").strip() for sent in context_sentences[-3:] if str(sent or "").strip()]
            sensory = self._contextual_sensory_bridge(recent, tone=tone)
            if sensory:
                return self._fit_char_window(sensory, min_chars, max_chars)
            micro_action = self._contextual_micro_action_bridge(recent, tone=tone)
            if micro_action:
                return self._fit_char_window(micro_action, min_chars, max_chars)
            for sent in reversed(recent):
                fragment = self._derive_clipped_bridge_fragment([sent])
                if fragment:
                    bridge = fragment if re.search(r"[.!?…]$", fragment) else f"{fragment}."
                    return self._fit_char_window(bridge, min_chars, max_chars)
            source = re.sub(r"\s+", " ", recent[-1]) if recent else ""
            source = re.sub(r"[.!?…]+$", "", source)
            if source:
                lead = re.split(r"[,，;—~]\s*", source, maxsplit=1)[0].strip()
                if len(re.findall(r"[0-9A-Za-z가-힣]+", lead)) >= 2:
                    if tone == "plain":
                        return self._fit_char_window(f"{lead}, 답이 바로 이어지지 않았다.", min_chars, max_chars)
                    return self._fit_char_window(f"{lead}, 손이 잠깐 멈췄다.", min_chars, max_chars)
        fallback = "답이 바로 이어지지 않았다." if tone == "plain" else "손이 잠깐 멈췄다."
        return self._fit_char_window(fallback, min_chars, max_chars)

    @staticmethod
    def _contextual_micro_action_bridge(sentences: list[str], tone: str = "strong") -> str:
        joined = " ".join(str(sent or "") for sent in sentences).lower()
        if not joined.strip():
            return ""
        if re.search(r"(질문|대답|응답|제안|조건|책임|통제|설명|협상|허가|대가)", joined):
            return "답이 바로 이어지지 않았다."
        if re.search(r"(경보|알림|alert|warning|monitor|비프)", joined):
            return "경보음에 손이 잠깐 멈췄다."
        if re.search(r"(문|복도|발소리|다가서|걸음|출입)", joined):
            return "시선이 문 쪽으로 옮겨갔다."
        if re.search(r"(시선|눈빛|응시)", joined):
            return "시선이 잠깐 흔들렸다."
        if re.search(r"(숨|호흡|정적|침묵|압박|긴장|불안|초조)", joined):
            return "숨이 짧게 멎었다."
        if re.search(r"(latency|coherence|drift|qpu|보정|오차|프로토콜|protocol)", joined):
            return "설명 대신 결과를 먼저 보게 됐다."
        return ""

    @staticmethod
    def _contextual_sensory_bridge(sentences: list[str], tone: str = "strong") -> str:
        joined = " ".join(str(sent or "") for sent in sentences).lower()
        if not joined.strip():
            return ""
        if re.search(r"(경보|알람|알림|alert|warning|monitor|비프|소리)", joined):
            return "경보음이 잠깐 더 선명해졌다."
        if re.search(r"(시선|응시|바라|눈)", joined):
            return "시선이 잠깐 멈췄다."
        if re.search(r"(심박|심장|숨|호흡|가쁘)", joined):
            return "숨이 한 번 얕아졌다."
        if re.search(r"(발소리|걸음|다가서|물러서|움직)", joined):
            return "발소리가 한 박자 먼저 닿았다."
        return "압박이 한 번 더 가까워졌다." if tone != "plain" else "다음 반응이 곧 이어졌다."

    @staticmethod
    def _sentence_word_count(sentence: str) -> int:
        return len(re.findall(r"[0-9A-Za-z가-힣]+", str(sentence or "")))

    def _strengthen_dialogue_action_beats(self, text: str) -> str:
        if not text:
            return text
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        cue_pattern = re.compile(r"(말했|물었|답했|중얼|속삭|고개|시선|손|입술|눈썹|웃|숨)")
        action_tags = [
            "그의 손끝이 테이블을 한 번 두드렸다.",
            "그녀가 시선을 비껴 짧게 숨을 골랐다.",
            "…말 사이로 공조기 소리만 낮게 흘렀다.",
            "누군가 컵 가장자리를 천천히 문질렀다.",
        ]
        action_idx = 0
        for block in blocks:
            sentences = self._split_korean_sentences(block)
            if not sentences:
                out_blocks.append(block)
                continue
            rebuilt: list[str] = []
            ambiguous_run = 0
            for sent in sentences:
                has_quote = bool(re.search(r"[\"“][^\"”\n]{3,}[\"”]", sent))
                has_cue = bool(cue_pattern.search(sent))
                rebuilt.append(sent)
                if has_quote and not has_cue:
                    ambiguous_run += 1
                else:
                    ambiguous_run = 0
                if ambiguous_run >= 2:
                    rebuilt.append(action_tags[action_idx % len(action_tags)])
                    action_idx += 1
                    ambiguous_run = 0
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _reinforce_name_refresh(self, text: str, protagonist_name: str) -> str:
        """
        If pronoun-only third-person references continue for several sentences,
        reinsert the protagonist name to reduce speaker/reference ambiguity.
        """
        if not text:
            return text
        constraints = self._feedback_style_constraints()
        raw = constraints.get("speaker_refresh_streak")
        try:
            streak_limit = int(raw)
        except (TypeError, ValueError):
            return text
        streak_limit = max(2, min(8, streak_limit))

        short_name = "수민" if "sumin" in protagonist_name.lower() else protagonist_name.strip()
        if not short_name:
            return text

        pronoun_pat = re.compile(r"^(그는|그가|그의|그를|그에게|그녀는|그녀가|그녀의|그녀를|그녀에게)\b")
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        for block in blocks:
            sentences = self._split_korean_sentences(block)
            if not sentences:
                out_blocks.append(block)
                continue
            run = 0
            rebuilt: list[str] = []
            for sent in sentences:
                cur = sent.strip()
                if not cur:
                    continue
                has_name = short_name in cur
                has_pronoun = bool(pronoun_pat.search(cur))
                if has_pronoun and not has_name:
                    run += 1
                else:
                    run = 0
                if run >= streak_limit and has_pronoun:
                    cur = pronoun_pat.sub(f"{short_name}은", cur, count=1)
                    run = 0
                rebuilt.append(cur)
            out_blocks.append(" ".join(rebuilt).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    def _apply_sensory_diversity_guard(self, text: str, recent_window: int = 3) -> str:
        if not text:
            return text
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        out_blocks: list[str] = []
        insert_idx = 0
        for block in blocks:
            sentences = self._split_korean_sentences(block)
            if not sentences:
                out_blocks.append(block)
                continue
            rebuilt: list[str] = []
            prev_channel = ""
            streak = 0
            for sent in sentences:
                channel = self._dominant_sensory_channel(sent)
                if channel and channel == prev_channel:
                    streak += 1
                elif channel:
                    prev_channel = channel
                    streak = 1
                rebuilt.append(sent)
                if channel and streak > recent_window:
                    rebuilt.append(self._sensory_switch_sentence(channel, insert_idx))
                    insert_idx += 1
                    streak = 0
                    prev_channel = ""
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())
        return "\n\n".join(b for b in out_blocks if b.strip())

    @staticmethod
    def _dominant_sensory_channel(sentence: str) -> str:
        low = str(sentence or "").lower()
        channel_tokens = {
            "visual": ("보였", "빛", "시선", "눈", "화면", "그림자", "반짝"),
            "sound": ("소리", "울림", "삐", "웅", "진동", "침묵", "속삭"),
            "touch": ("손끝", "피부", "거칠", "미끄", "닿", "떨림"),
            "temperature": ("차갑", "뜨겁", "열기", "냉기", "온기"),
            "smell": ("냄새", "향", "탄내", "금속 냄새"),
        }
        best = ""
        best_score = 0
        for channel, tokens in channel_tokens.items():
            score = sum(1 for t in tokens if t in low)
            if score > best_score:
                best = channel
                best_score = score
        return best

    @staticmethod
    def _sensory_switch_sentence(channel: str, idx: int) -> str:
        mapping = {
            "visual": [
                "공조기의 저음이 벽면을 타고 짧게 울렸다.",
                "손끝에는 차가운 금속의 결이 또렷하게 남았다.",
            ],
            "sound": [
                "공기가 식으며 피부 위 온도가 한 톤 내려갔다.",
                "희미한 금속 냄새가 코끝을 스쳤다.",
            ],
            "touch": [
                "형광등 빛이 테이블 모서리를 얇게 잘랐다.",
                "멀리서 끊긴 신호음이 다시 한 번 튕겼다.",
            ],
            "temperature": [
                "누군가 숨을 고르며 내는 작은 마찰음이 들렸다.",
                "손바닥에 닿은 표면이 예상보다 거칠었다.",
            ],
            "smell": [
                "표면 위 냉기가 손등으로 천천히 번졌다.",
                "멀리서 들린 발소리가 침묵을 깨뜨렸다.",
            ],
        }
        options = mapping.get(channel) or mapping["visual"]
        return options[idx % len(options)]

    def _log_paragraph_split_recommendations(self, text: str) -> None:
        if not text:
            return
        max_sentences = self._readability_controls()["paragraph_max"]
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        for idx, block in enumerate(blocks, start=1):
            sent_count = len(self._split_korean_sentences(block))
            if sent_count <= max_sentences:
                continue
            split_points = list(range(max_sentences, sent_count, max_sentences))
            logger.info(
                "Paragraph split recommendation #%d: %d sentences -> split near %s",
                idx,
                sent_count,
                split_points,
            )

    def _collect_style_diagnostics(self, text: str) -> dict[str, float]:
        if not text:
            return {
                "avg_paragraph_sentences": 0.0,
                "long_sentence_ratio": 0.0,
                "jargon_repeat_terms": 0.0,
                "max_visual_streak": 0.0,
                "transition_opener_repeats": 0.0,
                "pattern_warning_count": 0.0,
            }
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        sentence_counts = [len(self._split_korean_sentences(b)) for b in blocks if b]
        all_sentences = self._split_korean_sentences(text)
        long_sents = [s for s in all_sentences if self._sentence_word_count(s) >= 22]

        low = text.lower()
        jargon_terms = [e["variants"][0] for e in self._term_variation_catalog() if e.get("variants")]
        repeat_count = sum(1 for t in jargon_terms if self._count_feedback_term_occurrences(low, str(t)) >= 2)

        visual_tokens = ("보였", "빛", "시선", "눈", "화면", "그림자", "반짝")
        streak = 0
        max_streak = 0
        for sent in all_sentences:
            has_visual = any(tok in sent.lower() for tok in visual_tokens)
            if has_visual:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        transition_repeat_count = sum(
            1 for warning in self.detect_repetition_pattern_warnings(text)
            if "연결어" in warning or "시작 패턴" in warning
        )

        return {
            "avg_paragraph_sentences": (
                sum(sentence_counts) / max(len(sentence_counts), 1)
            ),
            "long_sentence_ratio": (
                len(long_sents) / max(len(all_sentences), 1)
            ),
            "jargon_repeat_terms": float(repeat_count),
            "max_visual_streak": float(max_streak),
            "transition_opener_repeats": float(transition_repeat_count),
            "pattern_warning_count": float(len(self.detect_repetition_pattern_warnings(text))),
        }

    def _warn_sensory_streak(self, text: str, streak_limit: int = 3) -> None:
        """
        Log a warning when same sensory channel (visual-heavy) repeats in streaks.
        """
        if not text:
            return
        visual_tokens = ("보였다", "빛", "시선", "눈", "화면", "그림자", "깜빡", "반짝")
        sentences = self._split_korean_sentences(text)
        streak = 0
        worst = 0
        for sent in sentences:
            low = sent.lower()
            has_visual = any(tok in low for tok in visual_tokens)
            if has_visual:
                streak += 1
                worst = max(worst, streak)
            else:
                streak = 0
        if worst >= streak_limit:
            logger.warning(
                "Sensory diversity warning: visual-channel streak reached %d consecutive sentences",
                worst,
            )

    @staticmethod
    def _split_long_sentence_soft(sentence: str, max_chars: int = 240) -> list[str]:
        """
        Soft-split one overly long sentence using Korean clause boundaries.
        Used only when punctuation-based sentence splitting yields one giant block.
        """
        s = str(sentence or "").strip()
        if not s:
            return []
        if len(s) <= max_chars:
            return [s]

        # Prefer clause separators that usually preserve semantic continuity.
        chunks = re.split(r"(?<=[,，;])\s+|(?<=\))\s+|(?<=다)\s+(?=그|하지만|그리고|그러나|다만|한편)", s)
        chunks = [c.strip() for c in chunks if c.strip()]
        if len(chunks) <= 1:
            tokens = s.split()
            out: list[str] = []
            buf: list[str] = []
            buf_chars = 0
            for tok in tokens:
                buf.append(tok)
                buf_chars += len(tok) + 1
                if buf_chars >= max_chars:
                    out.append(" ".join(buf).strip())
                    buf = []
                    buf_chars = 0
            if buf:
                out.append(" ".join(buf).strip())
            return out or [s]

        out: list[str] = []
        buf: list[str] = []
        buf_chars = 0
        for chunk in chunks:
            projected = buf_chars + len(chunk) + (1 if buf else 0)
            if buf and projected > max_chars:
                out.append(" ".join(buf).strip())
                buf = []
                buf_chars = 0
            buf.append(chunk)
            buf_chars += len(chunk) + 1
        if buf:
            out.append(" ".join(buf).strip())
        return out or [s]

    @staticmethod
    def _split_korean_sentences(block: str) -> list[str]:
        """
        Split Korean prose into sentence-like chunks.
        Uses punctuation first, then a soft-length fallback when punctuation is sparse.
        """
        text = str(block or "").strip()
        if not text:
            return []

        # Primary split: explicit sentence punctuation.
        by_punct = [
            s.strip()
            for s in re.split(r'(?<=[.!?…])\s+|(?<=[다요죠]\.)\s+', text)
            if s.strip()
        ]
        if len(by_punct) > 1:
            return by_punct

        # Fallback for long blocks with sparse punctuation.
        if len(text) <= 220:
            return [text]

        tokens = text.split()
        if len(tokens) < 12:
            return [text]

        chunks: list[str] = []
        buf: list[str] = []
        buf_chars = 0
        for tok in tokens:
            buf.append(tok)
            buf_chars += len(tok) + 1
            if buf_chars >= 120 and (tok.endswith(",") or tok.endswith(")") or buf_chars >= 170):
                chunks.append(" ".join(buf).strip())
                buf = []
                buf_chars = 0
        if buf:
            chunks.append(" ".join(buf).strip())

        return [c for c in chunks if c]

    # ------------------------------------------------------------------ #
    # Scene Word Budget
    # ------------------------------------------------------------------ #

    def _calculate_scene_budgets(
        self, scenes: list[DistilledScene], target_words: int
    ) -> list[int]:
        """Distribute word budget based on scene pacing."""
        if not scenes:
            return []

        weights = []
        for s in scenes:
            if s.pacing == "climax":
                weights.append(1.3)
            elif s.pacing in ("opening", "resolution"):
                weights.append(0.85)
            else:
                weights.append(1.0)

        total_weight = sum(weights)
        # Reserve ~15% for transitions
        prose_budget = int(target_words * 0.85)
        budgets = [int(prose_budget * w / total_weight) for w in weights]
        return budgets

    # ------------------------------------------------------------------ #
    # Title
    # ------------------------------------------------------------------ #

    def _generate_title(
        self, scenes: list[DistilledScene], episode_context: dict
    ) -> str:
        """Generate a literary chapter title."""
        summaries = " / ".join(s.title for s in scenes[:4])
        prompt = (
            f"Suggest a literary chapter title (3-8 words, Korean) for:\n"
            f"Episode {episode_context['episode_number']}: {episode_context['summary'][:200]}\n"
            f"Scenes: {summaries}\n"
            f"Reply with only the title, no quotes or punctuation."
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_title",
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
        scenes: list[DistilledScene],
    ) -> None:
        """Write the chapter Markdown file."""
        header = (
            f"# {title}\n\n"
            f"*Episode: {episode_id}*\n\n"
            f"---\n\n"
        )
        chapter_text = header + content

        # Reader-facing chapter should avoid replaying clue text verbatim.
        # Keep appendix optional for debugging/benchmark traceability only.
        if bool(self.runtime_policy.get("prose_append_debug_ledger", False)):
            scene_summary = "\n".join(
                f"  {i + 1}. {s.title} ({s.pacing})"
                for i, s in enumerate(scenes)
            )
            chapter_text += (
                f"\n\n---\n\n"
                f"*Scene structure:*\n{scene_summary}\n"
                f"\n"
                f"*Evidence ledger:*\n{self._build_evidence_ledger()}\n"
            )

        path.write_text(chapter_text, encoding="utf-8")

    def _build_evidence_ledger(self) -> str:
        """Compact clue ledger for traceability and fidelity checks."""
        clues = self.episode_config.get("introduced_clues", [])
        lines: list[str] = []
        for i, clue in enumerate(clues, start=1):
            if not isinstance(clue, dict):
                continue
            cid = str(clue.get("id", "")).strip() or f"clue_{i}"
            raw = str(clue.get("content", "")).strip()
            if not raw:
                continue
            compact = re.sub(r"\s+", " ", raw)
            lines.append(f"  - [{cid}] {compact}")
        return "\n".join(lines) if lines else "  - (none)"
