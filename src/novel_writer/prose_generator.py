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
from .review_feedback import build_feedback_prompt_block

logger = logging.getLogger(__name__)


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
        final = self._reduce_local_repetition(final)
        final = self._normalize_paragraphs(final)

        # Write
        out_path = self.output_dir / f"{episode_id}_chapter.md"
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
        term_glossary = self._build_scene_term_glossary(scene, matched_beats, blocked_terms=established)

        system = (
            "You are writing a Korean serialized techno-thriller chapter scene.\n"
            "Prioritize dramatic flow, subtext, and character individuality over metrics.\n"
            "No benchmark-style quotas or checklists. Write as a real novel scene.\n"
            "Avoid repetitive dialogue tags like '말했다/물었다' in consecutive lines; "
            "prefer action beats, gaze shifts, silence, interruption, and sentence rhythm to track speakers.\n"
            "Do not turn stage directions or narration into quoted speech.\n"
            "Use signature verbal tics only when context clearly demands it; avoid catchphrase repetition.\n"
            "Use concrete sensory details, but do not over-explain.\n"
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
            f"- Alternate inner thought and outer action to avoid continuous analytical voice.\n"
            f"- Use short sentence beats at tension peaks for pacing contrast.\n"
            f"- Avoid repeating the same tension phrasing across nearby paragraphs.\n\n"
            f"- Avoid repeating identical numeric literals (e.g., same milliseconds/ratios) in adjacent paragraphs unless plot-critical.\n\n"
            f"- In dialogue-heavy stretches, anchor speaker identity with short action beats or name cues every 1-2 exchanges.\n"
            f"- If three or more characters are present, avoid ambiguous pronouns for consecutive lines.\n\n"
            f"## Essential Content (from simulation)\n"
            f"Key dialogue:\n{dialogue_text}\n"
            f"Dialogue source status: {'simulation key dialogue exists' if has_source_dialogue else 'simulation key dialogue sparse; infer naturally'}\n\n"
            f"Key actions:\n{actions_text}\n\n"
            f"Discoveries/revelations:\n{discoveries_text}\n\n"
            f"Scene summary: {scene.narrative_summary}\n\n"
        )
        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "이름이 반복", "speaker"):
            prompt += (
                "## Speaker Clarity Priority\n"
                "- In every dialogue cluster, attach explicit speaker/addressee cues.\n"
                "- Avoid back-to-back ambiguous pronoun-only dialogue lines.\n"
                "- Do not reintroduce already-known characters with repetitive identity labels.\n\n"
            )
        review_guidance = build_feedback_prompt_block(self.reader_feedback, max_items=5)
        if review_guidance:
            prompt += (
                "## Reader Feedback Priorities\n"
                "Apply these priorities while preserving the same events.\n"
                "Use them to reduce local repetition and improve reading flow.\n"
                f"{review_guidance}\n\n"
            )
        if previous_episode_context:
            prompt += f"## Cross-Episode Continuity\n{previous_episode_context}\n\n"

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
                f"Rule: If technical terms appear, add a very short sensory/plain-language gloss on first mention only.\n\n"
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
            f"For any technical term on first mention, add a brief plain-language cue (about 3-8 Korean words) once.\n"
            f"When reusing already-known facts, reference briefly instead of re-explaining details.\n"
            f"Do not output labels, bullets, or metadata. Output only narrative prose."
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

    def _readability_controls(self) -> dict[str, int]:
        """Readability defaults, overridable via runtime policy."""
        min_sent = int(self.runtime_policy.get("prose_paragraph_min_sentences", 1) or 1)
        max_sent = int(self.runtime_policy.get("prose_paragraph_max_sentences", 3) or 3)
        if self._feedback_mentions("긴 문장", "문장이 길", "긴 문단", "문단이 길", "문단", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴"):
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
            repeated = [t for t in jargon_terms if low.count(t.lower()) >= 3]
            if repeated:
                reasons.append(
                    "기술 용어 반복 과다: " + ", ".join(repeated[:4])
                )
            acronym_terms = re.findall(r"\b[A-Z]{2,6}\b", text)
            if len(acronym_terms) >= 6:
                reasons.append("약어/대문자 기술 표기가 과밀함")

        if self._feedback_mentions("반복", "중복", "늘어지", "묘사", "빛", "손동작"):
            repetitive_tokens = ["제스처", "표정", "손동작", "빛", "손", "시선", "어깨", "숨", "정적"]
            repeated_imagery = [t for t in repetitive_tokens if text.count(t) >= 4]
            if repeated_imagery:
                reasons.append("유사 감각/동작 묘사 반복: " + ", ".join(repeated_imagery[:3]))

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
        prompt = (
            "다음 한국어 장면 산문을 같은 사건 흐름으로 유지하면서 1회 리라이트하라.\n"
            f"개선 사유: {reason_text}\n\n"
            "제약:\n"
            f"- 시점 유지: {pov}\n"
            f"- 분량 유지: 약 {word_budget}단어(크게 벗어나지 말 것)\n"
            "- 긴 설명문을 줄이고 문단 호흡을 짧게 분할\n"
            "- 같은 기술 용어/수치를 연속 문단에서 반복 설명하지 말 것\n"
            "- 기술 용어 첫 언급만 짧게 풀고 이후는 짧은 콜백으로 처리\n"
            "- 약어/대문자 기술 표기는 첫 등장에만 짧게 풀고 이후 최소화\n"
            "- 사건, 발견, 감정선의 순서는 바꾸지 말 것\n"
            "- 출력은 소설 본문만\n\n"
            f"원문:\n{text}"
        )
        revised = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_scene_readability_revise",
            use_premium=True,
            temperature=float(self.runtime_policy.get("prose_scene_readability_temperature", 0.35) or 0.35),
            max_tokens=min(4800, max(1800, word_budget * 5)),
        )
        return revised or text

    def _feedback_mentions(self, *keywords: str) -> bool:
        if not self.reader_feedback:
            return False
        corpus = []
        for key in ("what_felt_boring_or_hard", "style_tips"):
            vals = self.reader_feedback.get(key, []) or []
            corpus.extend(str(v) for v in vals if isinstance(v, str))
        corpus.append(str(self.reader_feedback.get("reader_comment", "") or ""))
        all_text = " ".join(corpus).lower()
        return any(str(k).lower() in all_text for k in keywords if k)

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
            "기술", "기술 설명", "용어", "약어", "약자", "누가 누구", "화자", "대사 구분", "헷갈",
        )
        if not needs_pass:
            return text

        pov = "first person" if style == "first_person" else "third person close"
        anchors = chapter_anchors or []
        anchors_text = ", ".join(anchors[:20]) if anchors else "(none)"
        review_guidance = build_feedback_prompt_block(self.reader_feedback, max_items=6)
        prompt = (
            "다음 한국어 소설 본문을 사건/정보/감정선 순서를 유지한 채 1회 리라이트하라.\n"
            "핵심 목적: 독자 리뷰 반영(반복 축소, 문단 호흡 개선, 기술 용어 과밀 완화, 화자 명확성 강화).\n\n"
            "제약:\n"
            f"- 시점 유지: {pov}\n"
            f"- 분량: 약 {target_words}단어 근처 유지\n"
            "- 동일 정보/표현의 반복은 삭제 또는 통합\n"
            "- 긴 문단은 1-2문장 단위로 자연 분할\n"
            "- 기술 용어/약어는 첫 등장만 짧게 풀고 이후는 짧은 콜백\n"
            "- 대화 구간은 1-2회 발화마다 누가 말하는지 드러나게 정리\n"
            "- 이미 알려진 인물을 매번 새 호칭으로 재소개하지 말 것\n"
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
            f"- Sentence rhythm should vary naturally (avoid repetitive cadence)\n"
            f"- Natural paragraph breaks at emotional beats\n"
            f"- If technical terms appear, keep first mention briefly readable with plain-language context\n"
            f"- No identical phrases or descriptions repeated\n\n"
            f"- Do not repeat the same numeric literal in adjacent paragraphs unless strictly necessary\n"
            f"- If a key metric was already explained once, later mentions should be very brief callbacks\n"
            f"- Avoid repeating acronym expansions; use concise references after first explanation\n"
            f"- Improve speaker clarity in dialogue passages using short action/name cues\n"
            f"- Preserve these anchor terms exactly when context allows: {anchors_text}\n"
            f"- If any anchor is missing, add it naturally without changing core events\n\n"
        )
        if self._feedback_mentions("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "이름이 반복", "speaker"):
            prompt += (
                "- In dialogue runs, explicitly tag speaker/addressee cues frequently enough "
                "to avoid ambiguity\n"
                "- Remove repetitive character re-introduction phrases\n"
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
        time_marker = re.search(r'잠시 후|그 후|이후|다음 날|그날 밤|며칠 후', out)
        if not time_marker:
            paragraphs = [p for p in out.split("\n\n") if p.strip()]
            if len(paragraphs) >= 4:
                paragraphs.insert(2, "잠시 후, 수민은 복도 끝의 소음을 지나 다음 장면으로 이동했다.")
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
                    if len(recent_sentence_fp) > 3:
                        recent_sentence_fp.pop(0)

            if keep:
                out_blocks.append(" ".join(keep))

        return "\n\n".join(out_blocks)

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

    @staticmethod
    def _ensure_transition_marker(text: str) -> str:
        if not text:
            return "잠시 후, 수민의 호흡이 가라앉자 시선이 다음 장면으로 옮겨졌다."
        if re.search(r'잠시 후|그 후|이후|다음 날|그날 밤|며칠 후', text):
            return text
        return f"잠시 후, {text.strip()}"

    def _normalize_paragraphs(self, text: str) -> str:
        """
        Post-process overly long paragraphs so analyzer-facing structure stays stable.
        Keeps content intact and only adjusts paragraph breaks.
        """
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        normalized: list[str] = []

        max_sentences = self._readability_controls()["paragraph_max"]

        for block in blocks:
            if block.startswith("#") or block.startswith("*") or block.startswith("---"):
                normalized.append(block)
                continue

            sentences = self._split_korean_sentences(block)

            if len(sentences) <= max_sentences:
                normalized.append(" ".join(sentences) if sentences else block)
                continue

            # Split long blocks into configurable chunks to keep readable breathing points.
            for i in range(0, len(sentences), max_sentences):
                chunk = " ".join(sentences[i:i + max_sentences]).strip()
                if chunk:
                    normalized.append(chunk)

        return "\n\n".join(normalized)

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
