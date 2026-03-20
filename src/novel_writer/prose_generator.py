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
        self.reader_feedback = reader_feedback or {}
        self.guardian_briefing = guardian_briefing or ""
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

        # Polish and guardrails
        final = self._polish(combined, target_words, style, coverage_anchors)
        final = self._ensure_anchor_coverage(final, coverage_anchors, target_words, style)
        final = self._reader_feedback_final_pass(final, target_words, style, coverage_anchors)
        final = self._enforce_pov_timeline_guards(final, style, protagonist_name)
        final = self._enforce_jargon_onboarding_and_variation(final)
        final = self._reduce_local_repetition(final)
        final = self._trim_redundant_sensory_sentences(final)
        final = self._trim_redundant_emotion_sentences(final)
        final = self._diversify_transition_openers(final)
        final = self._merge_clipped_sentence_runs(final)
        final = self._stagger_sentence_rhythm(final)
        final = self._compress_redundant_jargon_sentences(final)
        final = self._enforce_sentence_word_caps(final, max_words=self._feedback_sentence_word_cap(default=25))
        final = self._insert_short_beats_after_long_streak(
            final,
            long_threshold=22,
            streak_limit=2,
        )
        final = self._strengthen_dialogue_action_beats(final)
        final = self._split_dense_information_paragraphs(final)
        final = self._cap_paragraph_term_repetition(
            final,
            max_per_paragraph=self._feedback_term_repeat_cap(default=2),
        )
        final = self._apply_sensory_diversity_guard(final, recent_window=3)
        if style == "third_person_close":
            final = self._reinforce_name_refresh(final, protagonist_name)
        self._warn_sensory_streak(final, streak_limit=3)
        self._log_paragraph_split_recommendations(final)
        final = self._normalize_paragraphs(final)
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
            "Use concrete sensory details only at pressure turns; do not layer similar sensations repeatedly.\n"
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
            f"- Avoid repeating identical numeric literals (e.g., same milliseconds/ratios) in adjacent paragraphs unless plot-critical.\n\n"
            f"- In dialogue-heavy stretches, anchor speaker identity with short action beats or name cues every 1-2 exchanges.\n"
            f"- If three or more characters are present, avoid ambiguous pronouns for consecutive lines.\n\n"
            f"- Technical terms: explain only if the scene becomes unclear, and prefer brief inline cues over parentheses.\n"
            f"- Rotate sentence length in short/medium/long rhythm to avoid monotone cadence.\n"
            f"- If explanation runs long, pivot with one concrete action or reaction sentence instead of another detached fragment.\n"
            f"- If 2-3 short narrative sentences describe one beat, fuse them into one flowing sentence with clear cause/effect.\n"
            f"- Do not restate the same situation, emotion, or image in consecutive paragraphs unless new stakes changed it.\n"
            f"- When tension is already established, advance with one new choice, discovery, or reaction instead of repeating the same pressure line.\n"
            f"- Save the hardest tension wording for only one or two decisive beats in the scene.\n"
            f"- If similar sensory channel repeats for recent 3+ sentences, switch to another channel (sound/touch/temperature).\n"
            f"- Expository dialogue should be compressed; prioritize action/reaction beats after factual lines.\n\n"
            f"- Keep sensory description to about {sensory_cap} channels per paragraph, and save the sharpest image for the most important beat.\n"
            f"- Do not reuse the same emotion word or paraphrase more than about {emotion_repeat_cap} time per local beat.\n"
            f"- If an English keyword or technical term appears, explain it once in plain Korean in the next sentence, then attach an immediate human reaction or feeling.\n"
            f"- If commas or connectives start chaining clauses, cut the sentence into shorter direct beats.\n\n"
            f"- Do not lean on default connective openers like '그리고', '그러자', '다만' in nearby lines; vary or omit them when flow allows.\n"
            f"- Mix forceful pressure lines with plainer connective sentences so the rhythm rises and settles instead of staying equally taut.\n\n"
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
                "- Avoid repeating the same sentence-ending pattern across adjacent dialogue lines.\n\n"
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
            f"Each paragraph should add one fresh observation, decision, or emotional turn instead of circling the same tension.\n"
            f"When a technical or English term appears, make the very next sentence a plain-language explanation a high-school reader can follow.\n"
            f"Use sensory detail sparingly and only where it changes pressure, not as repeated atmosphere filler.\n"
            f"After that explanation, move straight to reaction, emotion, or decision.\n"
            f"Prefer short, direct sentences over comma-heavy chains when the beat turns sharp or explanatory.\n"
            f"Do not output labels, bullets, or metadata. Output only narrative prose."
        )
        prompt += (
            "\nAvoid repeating sentence openings with '그리고', '그러자', '다만'."
            "\nMix sharp emphasis lines with calmer connective lines so dialogue tension is not flat."
        )
        if self._feedback_mentions("심리", "내면", "설명적", "감정선", "표정", "행동", "보여"):
            prompt += (
                "\nUse action/gesture beats to externalize emotion before adding inner analysis."
            )
        if self._feedback_mentions("장면 전환", "전환", "복도", "발표장", "흐름"):
            prompt += (
                "\nAt each location/focus change, include a short transition sentence to keep flow explicit."
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
        if self._feedback_reports_stalled_progression():
            prompt += (
                "\nIf the same pressure already landed, jump to changed consequence in the very next sentence."
            )
        if self._feedback_mentions("반복되는 표현", "비슷한 상황", "비슷한 상황과 묘사", "묘사가 반복", "지루"):
            prompt += (
                "\nIf the same situation has already landed, keep one sharper callback and move to consequence."
            )
        if self._feedback_mentions("긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려"):
            prompt += (
                "\nIf dialogue runs long, break it with short action/reaction beats to protect pacing."
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
        return self._feedback_mentions(
            "멈춘 이유",
            "멈춘",
            "멈춤",
            "정체",
            "제자리",
            "안 나가",
            "진행이 안",
            "흐름이 끊",
        )

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
        if self._feedback_mentions("문장 구조", "반복적인 문장 구조", "비슷한 리듬", "같은 리듬", "단조", "지루"):
            if self._has_repetitive_sentence_openings(text):
                reasons.append("인접 문장의 시작 패턴이 반복되어 리듬이 단조로움")
            if self._has_transition_opener_streak(text):
                reasons.append("'그리고/그러자/다만' 계열 연결어 시작이 반복되어 흐름이 뻣뻣함")
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
        pov = "first person" if style == "first_person" else "third person close"
        reason_text = "; ".join(r for r in reasons if r).strip() or "가독성 개선 필요"
        sentence_cap = self._feedback_sentence_word_cap(default=25)
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
            "- 각 문단은 반드시 상황 변화, 압박, 발견 중 하나를 전진시킬 것\n"
            "- 어려운 기술 개념은 필요할 때만 짧은 일상 비유나 은유를 붙이고 바로 행동으로 돌아갈 것\n"
            "- 영어 키워드나 기술 용어 뒤에는 다음 문장으로 쉬운 풀어쓰기를 한 번 붙인 뒤, 곧바로 인물의 즉각적 반응, 감정, 판단을 붙일 것\n"
            "- 쉼표와 접속으로 절이 길게 이어지면 짧고 분명한 두 문장으로 나눌 것\n"
            "- 사건, 발견, 감정선의 순서는 바꾸지 말 것\n"
            "- 출력은 소설 본문만\n\n"
            f"원문:\n{text}"
        )
        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "인물", "역할", "구분", "호칭", "이름", "말투", "어투", "톤", "speaker"):
            prompt += (
                "\n추가 제약:\n"
                "- 대사 구간은 화자/청자 단서를 자주 배치해 혼선을 줄일 것\n"
                "- 동일 인물은 인접 문단에서 호칭을 과도하게 바꾸지 말 것\n"
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
        if any(k in lowered for k in ("긴 문장", "문장이 길", "긴 문단", "문단이 길", "문단", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴", "길고 복잡", "이해하기 어려", "이해하기 어렵")):
            if any(token in all_text for token in ("비슷한 리듬", "같은 리듬", "단조", "단조롭", "단조롭게", "리듬이 반복", "속도감이 단조", "속도감이 떨어", "템포가 느려", "템포가 떨어", "길고 복잡", "이해하기 어려", "이해하기 어렵")):
                return True

        # Reader may phrase jargon-density complaints as checklist/list-style repetition.
        if any(k in lowered for k in ("기술", "기술 설명", "용어", "약자", "약어", "전문", "jargon", "acronym")):
            if any(token in all_text for token in ("체크리스트", "나열", "리스트", "목록", "목록처럼", "긴 목록", "기술 항목", "건조", "단조롭")):
                return True

        if any(k in lowered for k in ("문장 구조", "반복적인 문장 구조", "비슷한 리듬", "같은 리듬", "단조", "지루", "반복되는 표현", "묘사가 반복")):
            if any(token in all_text for token in ("반복적인 문장 구조", "문장 구조가 반복", "비슷한 리듬", "같은 리듬", "단조", "지루", "반복되는 표현", "묘사가 반복")):
                return True

        if any(k in lowered for k in ("간결한 문장", "문맥 파악", "맥락 파악", "문맥", "맥락", "따라가기 힘들", "길고 복잡", "이해하기 어려", "이해하기 어렵")):
            if any(token in all_text for token in ("간결한 문장", "문맥 파악", "맥락 파악", "따라가기 힘들", "맥락이 약", "문맥이 약", "길고 복잡", "이해하기 어려", "이해하기 어렵")):
                return True

        if any(k in lowered for k in ("전개가 느려", "느려서 집중", "집중력을 잃", "늘어지", "템포가 느려", "속도감이 떨어")):
            if any(token in all_text for token in ("전개가 느려", "느려서 집중", "집중력을 잃", "집중력", "늘어지", "템포가 느려", "속도감이 떨어")):
                return True

        # Speaker/context confusion can appear as "초반에 따라가기 힘들다" style comments.
        if any(k in lowered for k in ("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "인물", "역할", "구분", "호칭", "이름", "speaker")):
            if any(token in all_text for token in ("초반", "따라가기 힘들", "맥락", "인물 설명 없이", "누군지")):
                return True

        # Expository dialogue pain can appear as "정보 전달 위주/감정의 고저 부족".
        if any(k in lowered for k in ("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트")):
            if any(token in all_text for token in ("정보 전달 위주", "설명 위주", "감정의 고저", "감정 고저", "대화가 대부분", "건조", "긴 회의", "회의·대화", "대화 장면", "대사가 계속")):
                return True

        # Emotional-wave requests often appear as "긴장 완화/유머/친근한 묘사/감정의 파고".
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
        return out[:10]

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
            if len(out) >= 10:
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
        # Rough conversion for Korean prose pacing (about 3.2 chars per token).
        inferred = max(10, min(default, int(round(hi / 3.2))))
        return inferred

    def _feedback_paragraph_sentence_cap(self) -> Optional[int]:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("max_sentences_per_paragraph")
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            return None
        return max(1, min(8, cap))

    def _feedback_dense_sentence_cap(self, default: int = 2) -> int:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("max_sentences_in_dense_info", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(4, cap))

    def _feedback_jargon_term_cap(self, default: int = 2) -> int:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("max_jargon_terms_per_paragraph", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(8, cap))

    def _feedback_sensory_channel_cap(self, default: int = 2) -> int:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("max_sensory_channels_per_paragraph", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(3, cap))

    def _feedback_emotion_repeat_cap(self, default: int = 1) -> int:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("max_emotion_repeats_per_scene", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(3, cap))

    def _feedback_transition_char_window(self) -> tuple[int, int]:
        constraints = self._feedback_style_constraints()
        try:
            lo = int(constraints.get("transition_chars_min", 10))
        except (TypeError, ValueError):
            lo = 10
        try:
            hi = int(constraints.get("transition_chars_max", 15))
        except (TypeError, ValueError):
            hi = 15
        if lo > hi:
            lo, hi = hi, lo
        lo = max(5, min(40, lo))
        hi = max(lo, min(40, hi))
        return lo, hi

    def _feedback_short_beat_char_window(self) -> tuple[int, int]:
        constraints = self._feedback_style_constraints()
        try:
            lo = int(constraints.get("short_beat_chars_min", 14))
        except (TypeError, ValueError):
            lo = 14
        try:
            hi = int(constraints.get("short_beat_chars_max", 28))
        except (TypeError, ValueError):
            hi = 28
        if lo > hi:
            lo, hi = hi, lo
        lo = max(8, min(32, lo))
        hi = max(lo, min(48, hi))
        return lo, hi

    def _feedback_short_beats_per_scene(self) -> tuple[int, int]:
        constraints = self._feedback_style_constraints()
        try:
            lo = int(constraints.get("short_beats_per_scene_min", 0))
        except (TypeError, ValueError):
            lo = 0
        try:
            hi = int(constraints.get("short_beats_per_scene_max", 1))
        except (TypeError, ValueError):
            hi = 1
        if lo > hi:
            lo, hi = hi, lo
        lo = max(0, min(8, lo))
        hi = max(lo, min(10, hi))
        return lo, hi

    def _feedback_transition_opener_cap(self, default: int = 2) -> int:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("max_transition_openers_per_block", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(4, cap))

    def _feedback_transition_avoid_terms(self) -> set[str]:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("avoid_transition_terms", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return set()
        return {
            re.sub(r"\s+", " ", str(term or "")).strip().lower()
            for term in raw
            if str(term or "").strip()
        }

    def _feedback_sentence_variety_window(self, default: int = 4) -> int:
        constraints = self._feedback_style_constraints()
        raw = constraints.get("sentence_variety_window", default)
        try:
            window = int(raw)
        except (TypeError, ValueError):
            window = default
        return max(3, min(6, window))

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
        """
        Final low-temperature pass that explicitly applies reader feedback.
        This catches residual repetition/speaker-clarity issues after normal polish.
        """
        if not text or not self.reader_feedback:
            return text

        needs_pass = self._feedback_mentions(
            "반복", "중복", "늘어지", "긴 문장", "문장이 길", "긴 문단", "문단이 길",
            "문장 구조", "반복적인 문장 구조", "간결한 문장", "문맥 파악", "맥락 파악",
            "전개가 느려", "느려서 집중", "집중력을 잃",
            "기술", "기술 설명", "용어", "약어", "약자", "누가 누구", "화자", "대사 구분", "헷갈",
            "인물", "역할", "구분", "호칭", "이름",
            "심리", "심리 표현", "내면", "설명적", "감정선", "감정 표현", "표정", "행동", "보여", "장면 전환", "전환", "흐름",
            "체크리스트", "나열", "초반", "따라가기 힘들",
            "긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려",
            "감정의 고저", "감정 고저", "감정의 파고", "긴장 완화", "유머", "친근한 묘사",
            "비슷한 감각 묘사", "감각 묘사", "영어 키워드", "영어 표현", "해석 부담",
        )
        if not needs_pass:
            return text

        pov = "first person" if style == "first_person" else "third person close"
        anchors = chapter_anchors or []
        anchors_text = ", ".join(anchors[:20]) if anchors else "(none)"
        review_guidance = build_feedback_prompt_block(self.reader_feedback, max_items=6)
        repeat_terms = self._feedback_repeat_terms()
        jargon_terms = self._feedback_jargon_terms()
        repeat_term_line = (
            f"- 독자 반복 지적 단어({', '.join(repeat_terms[:6])})는 문단당 과다 반복 금지\n"
            if repeat_terms else ""
        )
        jargon_term_line = (
            f"- 독자 난해 지적 기술어({', '.join(jargon_terms[:6])})는 첫 언급에만 짧게 풀고 재등장은 축약\n"
            if jargon_terms else ""
        )
        jargon_density_cap = self._feedback_jargon_term_cap(default=2)
        dense_sentence_cap = self._feedback_dense_sentence_cap(default=2)
        paragraph_sentence_cap = self._feedback_paragraph_sentence_cap()
        sentence_cap = self._feedback_sentence_word_cap(default=25)
        paragraph_cap_line = (
            f"- 문단은 최대 {paragraph_sentence_cap}문장까지 유지하고 초과 시 분할\n"
            if paragraph_sentence_cap is not None else ""
        )
        prompt = (
            "다음 한국어 소설 본문을 사건/정보/감정선 순서를 유지한 채 1회 리라이트하라.\n"
            "핵심 목적: 독자 리뷰 반영(반복 축소, 문단 호흡 개선, 기술 용어 과밀 완화, 화자 명확성 강화).\n\n"
            "제약:\n"
            f"- 시점 유지: {pov}\n"
            f"- 분량: 약 {target_words}단어 근처 유지\n"
            "- 동일 정보/표현의 반복은 삭제 또는 통합\n"
            "- 긴 문단은 1-2문장 단위로 자연 분할\n"
            f"{paragraph_cap_line}"
            f"- 대부분의 문장은 약 {sentence_cap}어절 이하로 유지하고, 길어지면 인과 단위로 분리\n"
            "- 같은 문장 구조나 문장 시작 패턴이 이어지면 하나 이상 변형해 리듬을 바꿀 것\n"
            "- '그리고', '그러자', '다만' 같은 연결어를 연속 문장 시작에 반복하지 말 것\n"
            "- 짧은 문장이 연속될 때는 누가/왜/어디서가 보이도록 연결해 문맥을 보강할 것\n"
            "- 같은 장면의 2~3개 단문이 한 박자로 이어지면 하나의 복합문으로 자연스럽게 묶을 것\n"
            "- 강한 문장과 담백한 문장을 섞어 압박의 고저를 만들 것\n"
            "- 기술 용어/약어는 꼭 필요할 때만 짧게 풀고 이후는 짧은 콜백으로 유지할 것\n"
            "- 괄호 설명은 기본값으로 쓰지 말고, 필요하면 본문 안에 짧게 녹여 쓸 것\n"
            "- 어려운 기술 개념은 필요할 때만 짧은 일상 비교를 한 번 붙이고 바로 장면 행동으로 돌아갈 것\n"
            f"- 문단당 기술 용어는 최대 {jargon_density_cap}개 내에서 유지(초과 개념은 통합/요약)\n"
            f"- 정보량이 많은 설명 문장은 최대 {dense_sentence_cap}문장으로 압축\n"
            f"{repeat_term_line}"
            f"{jargon_term_line}"
            "- 대화 구간은 1-2회 발화마다 누가 말하는지 드러나게 정리\n"
            "- 정보 전달형 대사는 짧게 압축하고, 바로 행동/표정/침묵 반응을 붙여 임팩트를 살릴 것\n"
            "- 긴 회의/대화 구간은 연속 설명 대사를 줄이고 행동/환경 반응 비트를 교차 배치할 것\n"
            "- 설명적 심리문이 길면 행동/표정/반응 단서로 치환해 감정을 보여줄 것\n"
            "- 비슷한 감각 묘사와 심리 표현은 같은 문단/인접 문단에서 반복하지 말 것\n"
            "- 감정 강도는 단조롭게 유지하지 말고 짧은 완화 비트 후 다시 긴장을 세울 것\n"
            "- 각 문단은 상황 변화, 압박 상승, 발견 중 하나를 분명히 남겨 전개를 전진시킬 것\n"
            "- 장소/장면 전환 지점은 한 줄 전환 문장으로 연결해 흐름을 명확히 할 것\n"
            "- 가능성/계산/추론 같은 분석 어휘는 반복하지 말고 한 번만 압축적으로 사용\n"
            "- 이미 알려진 인물을 매번 새 호칭으로 재소개하지 말 것\n"
            "- 인물의 역할/의도는 장면상 필요할 때만 짧게 제시하고 중복 설명은 삭제\n"
            "- 영어 키워드/기술 용어 뒤에는 바로 인물의 이해, 당혹, 긴장, 행동 반응을 붙일 것\n"
            f"- 가능한 맥락에서 다음 앵커를 보존: {anchors_text}\n"
            "- 출력은 소설 본문만\n\n"
            "독자 피드백:\n"
            f"{review_guidance}\n\n"
            f"원문:\n{text}"
        )
        revised = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_reader_feedback_pass",
            use_premium=True,
            temperature=float(self.runtime_policy.get("prose_reader_feedback_temperature", 0.25) or 0.25),
            max_tokens=min(16000, max(6000, target_words * 5)),
        )
        return revised or text

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
            repeat_hits = len(re.findall(r"(생각|계산|판단|설명|반복|다시)", low))
            if change_hits == 0 and action_hits <= 1 and repeat_hits >= 2:
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
        strict_jargon_control = self._feedback_mentions(
            "기술", "용어", "약자", "약어", "전문", "jargon", "acronym", "반복", "중복"
        )
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
        """Final consistency and word count pass."""
        current = len(text.split())
        pov = "first person" if style == "first_person" else "third person close"
        anchors = chapter_anchors or []
        anchors_text = ", ".join(anchors[:30]) if anchors else "(none)"

        if current < target_words * 0.7:
            instruction = (
                f"The chapter is {current} words but should be ~{target_words}. "
                f"Expand with additional sensory detail, deeper internal reflection, "
                f"and richer scene-setting. Do NOT add new plot events."
            )
        elif current > target_words * 1.4:
            instruction = (
                f"The chapter is {current} words but should be ~{target_words}. "
                f"Tighten the prose: remove repetition, merge redundant descriptions, "
                f"cut filler phrases. Preserve all key events and dialogue."
            )
        else:
            instruction = (
                f"The chapter is {current} words (target: ~{target_words}). "
                f"Do a final review for: consistent {pov} voice, smooth flow, "
                f"no abrupt tonal shifts. Make only minor improvements."
            )

        prompt = (
            f"{instruction}\n\n"
            f"Also ensure:\n"
            f"- Consistent {pov} voice throughout (Korean)\n"
            f"- No simulation artifacts (turn numbers, metadata, labels)\n"
            f"- Paragraphs should usually contain {self._readability_controls()['paragraph_min']}-{self._readability_controls()['paragraph_max']} sentences\n"
            f"- Most sentences should stay under about {self._feedback_sentence_word_cap(default=25)} words; split explanatory chains early\n"
            f"- Sentence rhythm should vary naturally (avoid repetitive cadence)\n"
            f"- Natural paragraph breaks at emotional beats\n"
            f"- If technical terms appear, keep first mention briefly readable with plain-language context\n"
            f"- No identical phrases or descriptions repeated\n\n"
            f"- Do not repeat the same numeric literal in adjacent paragraphs unless strictly necessary\n"
            f"- If a key metric was already explained once, later mentions should be very brief callbacks\n"
            f"- Avoid repeating acronym expansions; use concise references after first explanation\n"
            f"- On first mention of a technical term/acronym, use one short inline cue only if clarity truly needs it\n"
            f"- If dense technical info appears, split into short sentences or short beat-style line breaks\n"
            f"- Improve speaker clarity in dialogue passages using short action/name cues\n"
            f"- If a concept recurs (coherence/drift/latency classes), vary surface wording while keeping meaning stable\n"
            f"- If 3+ consecutive sentences use same sensory channel, switch channel (sound/touch/temperature)\n"
            f"- If explanatory rhythm grows monotonous, use one grounded action/reaction sentence instead of a detached fragment\n"
            f"- Preserve these anchor terms exactly when context allows: {anchors_text}\n"
            f"- If any anchor is missing, add it naturally without changing core events\n\n"
        )
        if self._feedback_mentions("동의어", "통일", "의미 중복", "혼선"):
            prompt += (
                "- Keep one stable term per concept; avoid synonym swapping for the same idea\n"
            )
        if self._feedback_mentions("정보가 많은 단락", "정보 밀집", "요약 문장", "핵심을 정리", "핵심 정리"):
            prompt += (
                "- End dense information paragraphs with one short takeaway summary sentence\n"
            )
        if self._feedback_mentions("감각 묘사", "감각", "선명도", "1~2"):
            prompt += (
                "- Keep sensory detail focused to 1-2 sensory channels per paragraph\n"
            )
        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "이름이 반복", "인물", "역할", "구분", "호칭", "이름", "speaker"):
            prompt += (
                "- In dialogue runs, explicitly tag speaker/addressee cues frequently enough "
                "to avoid ambiguity\n"
                "- Keep character naming stable and remove repetitive re-introduction phrases\n"
            )
        review_guidance = build_feedback_prompt_block(self.reader_feedback, max_items=5)
        if review_guidance:
            prompt += (
                "Additional reader feedback to honor during polish:\n"
                f"{review_guidance}\n\n"
            )
        prompt += (
            f"Full chapter text:\n\n{text}"
        )

        polished = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_polish",
            use_premium=True,
            temperature=float(self.runtime_policy.get("prose_polish_temperature", 0.4) or 0.4),
            max_tokens=min(16000, max(6000, target_words * 5)),
        )
        return self._normalize_paragraphs(polished)

    def _ensure_anchor_coverage(
        self,
        text: str,
        chapter_anchors: list[str],
        target_words: int,
        style: str,
    ) -> str:
        """
        Final guardrail: if anchor coverage is weak, revise once to include
        missing evidence terms naturally without changing plot events.
        """
        if not text or not chapter_anchors:
            return text

        def has_anchor(src: str, anchor: str) -> bool:
            return anchor.lower() in src.lower()

        anchors = [a.strip() for a in chapter_anchors if isinstance(a, str) and len(a.strip()) >= 3][:30]
        if not anchors:
            return text

        present = [a for a in anchors if has_anchor(text, a)]
        # Reasonable floor across episodes; only trigger when clearly under-covered.
        required_present = min(5, max(2, len(anchors) // 5))
        if self._feedback_mentions("기술", "용어", "약자", "약어", "전문", "jargon", "acronym", "반복", "중복"):
            # Prioritize readability when reviews repeatedly complain about jargon/repetition.
            required_present = min(required_present, max(1, len(anchors) // 6))
        if target_words < 2200:
            required_present = min(required_present, 2)
        if len(present) >= required_present:
            return text

        missing_cap = 3 if self._feedback_mentions(
            "기술", "용어", "약자", "약어", "전문", "jargon", "acronym", "반복", "중복"
        ) else 6
        missing = [a for a in anchors if a not in present][:missing_cap]
        pov = "first person" if style == "first_person" else "third person close"
        prompt = (
            f"Revise this Korean chapter to preserve story flow while increasing evidence fidelity.\n\n"
            f"Hard constraints:\n"
            f"- Keep the same events and scene order.\n"
            f"- Keep {pov} voice.\n"
            f"- Keep total length near {target_words} words.\n"
            f"- Integrate these missing anchor terms verbatim and naturally:\n"
            f"  {', '.join(missing)}\n\n"
            f"- Do not repeat full technical explanations; use short callbacks if already introduced.\n"
            f"Return only revised chapter text.\n\n"
            f"Chapter:\n{text}"
        )
        revised = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_anchor_fix",
            use_premium=True,
            temperature=float(self.runtime_policy.get("prose_anchor_fix_temperature", 0.35) or 0.35),
            max_tokens=min(16000, max(6000, target_words * 5)),
        )
        return self._normalize_paragraphs(revised)

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
                    "이윽고, 장면의 공기가 미세하게 바뀌었다.",
                    "그사이, 시선은 다음 움직임으로 천천히 옮겨갔다.",
                    "곧이어, 긴장선은 다른 지점으로 이어졌다.",
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
                out_blocks.append(" ".join(keep))

        return "\n\n".join(out_blocks)

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
                    if replacement:
                        pattern = re.escape(connector)
                        sent = re.sub(
                            rf"^([\"“”'‘’\(\)\[\]\s]*){pattern}\s*,?\s*",
                            rf"\1{replacement} ",
                            sent.strip(),
                            count=1,
                        ).strip()
                        connector = replacement
                        connector_key = connector.lower()
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
            "그리고": ["그러자", "그 직후", "잠시 뒤", "한편"],
            "그러자": ["그리고", "그러는 사이", "그 말이 끝나자", "곧바로"],
            "다만": ["대신", "문제는", "그래도", "한편"],
            "이어서": ["그리고", "그 직후", "그러는 사이", "곧바로"],
            "그 순간": ["그러자", "바로 그때", "그 말이 끝나자", "순간적으로"],
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
        This directly targets the clipped, mechanical cadence flagged in reviews.
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
                if len(run) >= 2:
                    rebuilt.append(self._combine_clipped_sentence_run(run))
                else:
                    rebuilt.extend(run)
                run = []
                rebuilt.append(sent.strip())
            if len(run) >= 2:
                rebuilt.append(self._combine_clipped_sentence_run(run))
            else:
                rebuilt.extend(run)
            out_blocks.append(" ".join(s for s in rebuilt if s.strip()).strip())

        return "\n\n".join(b for b in out_blocks if b.strip())

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

    def _combine_clipped_sentence_run(self, sentences: list[str]) -> str:
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0]
        connectors = [
            conn for conn in ["그리고", "그러자", "그러는 사이", "그 직후"]
            if conn.lower() not in self._feedback_transition_avoid_terms()
        ]
        if not connectors:
            connectors = ["그리고", "그러자"]
        combined = re.sub(r"[.!?…]+$", "", sentences[0].strip())
        for idx, sent in enumerate(sentences[1:], start=1):
            cleaned = re.sub(r"[.!?…]+$", "", sent.strip())
            if not cleaned:
                continue
            connector = connectors[(idx - 1) % len(connectors)]
            combined = f"{combined}, {connector} {cleaned}"
        return combined.strip(" ,") + "."

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
        fallback = "곧이어 공기가 변했다."
        if not text:
            return self._fit_char_window(fallback, lo, hi)
        out = text.strip()
        if not re.search(r'잠시 후|그 후|이후|그사이|한편|이윽고|곧이어|다음 날|그날 밤|며칠 후', out):
            variants = ("잠시 후,", "그사이,", "이윽고,", "곧이어,")
            idx = sum(ord(ch) for ch in out) % len(variants)
            out = f"{variants[idx]} {out}"
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
        return out

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
                    rebuilt.append(self._rhythm_bridge_sentence(beat_idx, min_short, max_short, tone=tone))
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
    ) -> str:
        min_chars = max(min_chars, 14)
        max_chars = max(max_chars, 28)
        strong_samples = [
            "질문의 방향이 바뀌자 테이블 반대편의 시선도 함께 움직였다.",
            "의자가 살짝 밀리는 소리 뒤로 회의장의 호흡이 한 박자 늦어졌다.",
            "누군가 메모를 멈추고 고개를 들면서 방 안의 집중이 한곳으로 쏠렸다.",
            "목 안쪽이 마르게 조여 오자 수민은 다음 말을 더 천천히 골랐다.",
            "말끝이 한쪽으로 기울자 상대의 반응도 노골적으로 달라졌다.",
            "공조기 소리만 남은 틈에서 누구도 먼저 시선을 거두지 않았다.",
            "테이블 위 그림자가 흔들리자 분위기 역시 미세하게 균형을 잃었다.",
            "한 사람이 먼저 숨을 고르자 다른 이들도 그 변화를 놓치지 못했다.",
            "이번에는 시선이 정면으로 맞붙었고, 피할 구실은 더 이상 남지 않았다.",
            "종이 모서리가 천천히 접히는 동안 아무도 성급하게 답을 내놓지 않았다.",
            "답이 늦어지는 사이 방 안에는 계산보다 긴장이 먼저 차올랐다.",
            "손끝에 들어간 힘이 굳어지자 질문의 무게도 분명하게 드러났다.",
            "누군가 미소를 접는 순간, 감춰 두던 의도 역시 절반쯤 모습을 드러냈다.",
            "질서 있던 호흡이 한 박자 어긋나면서 대화의 결도 서서히 거칠어졌다.",
            "단어 하나가 분위기를 바꾸자 모두가 그 다음 문장을 기다렸다.",
            "정적이 오래 끌리진 않았지만, 그 짧은 틈은 이미 충분한 의미를 남겼다.",
            "기계음이 박자를 끊는 사이 사람들의 판단도 잠깐씩 서로를 엇갈렸다.",
            "고개를 든 사람은 한 명뿐이었고, 그 사실이 방 안의 균형을 바꿨다.",
            "실내의 온도가 내려간 듯한 착각과 함께 반응의 방향도 선명해졌다.",
            "그제야 모두가, 지금부터는 말보다 선택이 더 중요해졌다는 걸 알아차렸다.",
        ]
        plain_samples = [
            "수민은 바로 대답하지 않았다.",
            "누군가 메모를 한 줄 더 적었다.",
            "컵 바닥이 조용히 테이블에 닿았다.",
            "그는 숨을 한 번 고르고 말을 골랐다.",
            "짧은 정리 뒤에 다음 질문이 이어졌다.",
            "말의 끝이 가라앉자 방 안도 잠깐 조용해졌다.",
            "시선 몇 개가 같은 지점에 모였다가 다시 흩어졌다.",
            "누군가는 고개만 아주 작게 끄덕였다.",
        ]
        samples = plain_samples if tone == "plain" else strong_samples
        sample = samples[idx % len(samples)]
        return self._fit_char_window(sample, min_chars, max_chars)

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
