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

from .llm_client import LLMClient
from . import database as db
from .review_feedback import build_feedback_prompt_block

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
        runtime_policy: Optional[dict] = None,
        reader_feedback: Optional[dict] = None,
    ) -> None:
        self.llm = llm
        self.episode_config = episode_config or {}
        self.runtime_policy = runtime_policy or {}
        self.reader_feedback = reader_feedback or {}

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
        compact_beats = self._reader_prefers_compact_beats()
        for c in clues:
            if isinstance(c, dict):
                raw_content = str(c.get("content", "") or "")
                beats.append({
                    "id": c.get("id", ""),
                    "content": self._compress_beat_content(raw_content) if compact_beats else raw_content,
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
        canonical_speakers = self._build_canonical_speaker_map(interactions)
        # Format interactions compactly
        turns_text = self._format_turns_compact(interactions)

        # Format beats for reference
        beats_text = "\n".join(
            f"- [{b['id']}]: {b['content']}" for b in beats
        ) or "(no beats defined)"

        ep_summary = self.episode_config.get("summary", "")
        ep_location = self.episode_config.get("location", "")
        ep_pacing = self.episode_config.get("pacing", "normal")
        summary_word_cap = self._summary_sentence_word_cap(default=18)

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
            f"6. Compress explanatory dialogue: keep one decisive quote, convert the rest into action/reaction summary.\n"
            f"7. Keep technical-term onboarding compact: first clear mention only, later references summarized.\n"
            f"8. Keep summary rhythm natural: mostly clear medium-length sentences, with a short sentence only when a real turn in pressure needs emphasis.\n"
            f"9. Make each summary easy to follow: name the acting subject early and keep cause/effect explicit instead of stacking clipped fragments.\n"
            f"10. Do not preserve a scene unless it changes tension, information, or decision; merge low-movement beats into the adjacent scene summary.\n"
            f"11. Do NOT invent content not present in the log. Only compress and select.\n\n"
            f"12. Prefer 2 clear summary sentences; use a third only when the scene includes a distinct reversal or discovery, and keep them as complete sentences rather than fragments.\n"
            f"13. If adjacent candidate scenes share cast/location and the latter mainly restates mood or explanation, merge them.\n"
            f"14. Remove repeated atmosphere, gesture, or technical explanation phrasing unless stakes visibly change.\n\n"
            f"15. Keep mood-only fragments such as silence/noise/gesture cues only when they mark an actual turn in pressure; otherwise rewrite them as action or emotional consequence in the same sentence.\n"
            f"16. If a summary sentence is mostly atmosphere, pair it with who moved, decided, or reacted so the scene does not feel frozen.\n\n"
            f"17. If a technical term or acronym survives in the summary, add one short plain-language Korean sentence immediately after it, then show an immediate human reaction or consequence.\n"
            f"18. When abstraction rises, ground it in what the character instantly felt, heard, saw, or did.\n"
            f"19. Replace abstract or lofty phrasing with direct cause-and-effect wording that a high-school reader can follow quickly.\n"
            f"20. Keep each summary sentence under about {summary_word_cap} words. If commas/connectives start chaining clauses, split them into shorter sentences.\n"
            f"21. Avoid stock sentence-leading connectives like '그리고', '그러자', '다만' unless a real turn, interruption, or location shift happens there.\n"
            f"22. If two candidate lines restate the same pressure or explanation, keep the line with clearer action/reaction and drop the other.\n"
            f"23. When scene count is limited, preserve higher-tension beats first: reveal, interruption, exposed contradiction, forced decision, or visible stress reaction.\n"
            f"24. If external support, real-time control, resources, authority, or responsibility repeat across late turns, keep the clearest first mention and compress later restatements into changed consequence only.\n"
            f"25. Treat transition-making beats one at a time: memo discovery, warning sound, and named arrival should each land in their own clear sentence with who noticed it and where they were.\n"
            f"26. If an unnamed observer later appears by name, either link the identity once or keep the vague observer clearly separate; do not alternate between '정장 남자' and a proper name without clarification.\n\n"
        )
        review_guidance = build_feedback_prompt_block(self.reader_feedback, max_items=5)
        if review_guidance:
            prompt += (
                "## Reader Feedback Priorities\n"
                "Favor readability when selecting what survives compression.\n"
                "If technical explanations repeat, keep only the clearest first mention.\n"
                "Prefer concise summaries over long duplicate emotional narration.\n"
                "Keep dialogue attribution explicit; avoid preserving multiple near-identical lines "
                "from the same speaker.\n"
                f"{review_guidance}\n\n"
            )
        repeat_terms = self.reader_feedback.get("repetition_watch_terms", []) or []
        cleaned_repeat_terms = []
        generic_repeat_terms = {"묘사", "표현", "설명", "감정", "분위기", "정보"}
        for raw in repeat_terms:
            term = re.sub(
                r"\s+(반복|중복|과다|과잉|묘사|표현)$",
                "",
                str(raw or "").strip(),
                flags=re.IGNORECASE,
            ).strip()
            if term and term not in generic_repeat_terms:
                cleaned_repeat_terms.append(term)
        if cleaned_repeat_terms:
            prompt += (
                "## Repetition Watch Terms\n"
                "- Reader flagged these motif words as repetitive in recent drafts.\n"
                f"- Terms: {', '.join(cleaned_repeat_terms[:6])}\n"
                "- Preserve each term at most once unless a beat explicitly requires it.\n\n"
            )
        jargon_terms = self.reader_feedback.get("jargon_watch_terms", []) or []
        cleaned_jargon_terms = []
        for raw in jargon_terms:
            term = str(raw or "").strip()
            term = term.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
            term = re.sub(r"\s+", " ", term).strip()
            if term:
                cleaned_jargon_terms.append(term)
        if cleaned_jargon_terms:
            prompt += (
                "## Jargon Watch Terms\n"
                "- Reader reported these terms as hard to follow when repeated without context.\n"
                f"- Terms: {', '.join(cleaned_jargon_terms[:6])}\n"
                "- If needed, preserve only one clear first mention and summarize later references.\n\n"
            )
        constraints = self.reader_feedback.get("style_constraints", {}) if self.reader_feedback else {}
        if isinstance(constraints, dict) and constraints:
            lines: list[str] = []
            para_sent_cap = constraints.get("max_sentences_per_paragraph")
            jargon_cap = constraints.get("max_jargon_terms_per_paragraph")
            dense_cap = constraints.get("max_sentences_in_dense_info")
            summary_word_cap_cfg = constraints.get("scene_summary_sentence_words_max")
            if isinstance(para_sent_cap, int):
                lines.append(f"- Keep scene summary paragraphs under about {para_sent_cap} sentence(s).")
            if isinstance(jargon_cap, int):
                lines.append(f"- Keep technical/jargon concepts to about {jargon_cap} per paragraph.")
            if isinstance(dense_cap, int):
                lines.append(f"- Compress dense explanatory runs into <= {dense_cap} sentence chunks.")
            if isinstance(summary_word_cap_cfg, int):
                lines.append(f"- Keep each scene-summary sentence to about {summary_word_cap_cfg} words or less.")
            if lines:
                prompt += "## Numeric Style Constraints\n" + "\n".join(lines) + "\n\n"
        feedback_corpus = " ".join(
            str(x)
            for key in ("what_felt_boring_or_hard", "style_tips", "reader_comment")
            for x in (
                self.reader_feedback.get(key, [])
                if isinstance(self.reader_feedback.get(key, []), list)
                else [self.reader_feedback.get(key, "")]
            )
            if str(x).strip()
        ).lower()
        if any(k in feedback_corpus for k in ("장면 전환", "전환", "복도", "발표장", "흐름")):
            prompt += (
                "When location/time shifts, keep one short transition cue sentence in scene summary "
                "(example shape: '수민은 복도로 나섰다.').\n\n"
            )
        if any(k in feedback_corpus for k in ("체크리스트", "나열", "리스트", "목록", "목록처럼", "긴 목록", "기술 항목", "단조", "건조")):
            prompt += (
                "If source turns include checklist/list-like technical explanation strings, "
                "compress them into one concise action consequence sentence instead of preserving list form.\n\n"
            )
        if any(k in feedback_corpus for k in ("비슷한 리듬", "같은 리듬", "단조", "단조롭", "속도감이 단조")):
            prompt += (
                "Vary sentence length inside summaries, but keep the default as full natural sentences rather than repeated clipped beats.\n\n"
            )
        if any(k in feedback_corpus for k in ("반복되는 표현", "비슷한 상황", "비슷한 상황과 묘사", "묘사가 반복", "지루")):
            prompt += (
                "If two nearby moments express the same tension with similar wording, keep only the sharper phrasing once and turn the rest into consequence.\n\n"
            )
        if any(k in feedback_corpus for k in ("문장이 너무 길", "길고 복잡", "이해하기 어려", "이해하기 어렵")):
            prompt += (
                "Keep summaries syntactically simple: one action or discovery per sentence, with explicit subject early in the line.\n\n"
            )
        if any(k in feedback_corpus for k in ("긴 회의", "회의·대화", "대화 장면", "속도감이 떨어", "템포가 느려")):
            prompt += (
                "When dialogue stretches long, keep only one decisive quote and summarize the rest as action/reaction.\n"
                "This prevents mid-scene pacing drops.\n\n"
            )
        if self._reader_reports_stalled_progression():
            prompt += (
                "Reader flagged that the chapter feels stalled.\n"
                "Merge pauses, repeated analysis, and lingering mood beats into the nearest scene unless they create a real decision, discovery, or pressure change.\n"
                "End each retained scene summary on what changed next, not on atmosphere alone.\n\n"
            )
        if any(k in feedback_corpus for k in ("심리", "내면", "설명적", "감정선", "표정", "행동", "보여")):
            prompt += (
                "Prefer observable emotion evidence (micro-action, gaze, posture) over abstract "
                "psychological explanation in summaries.\n\n"
            )
        if any(k in feedback_corpus for k in ("감정의 파고", "감정 파고", "감정의 고저", "감정 고저", "긴장 완화", "작은 유머", "유머", "친근한 묘사")):
            prompt += (
                "Across consecutive scenes, keep emotional wave contrast visible "
                "(e.g., one brief easing beat before renewed tension).\n\n"
            )
        if any(k in feedback_corpus for k in ("인물", "이름", "직책", "역할", "구분", "헷갈")):
            prompt += (
                "When many characters appear, add one short role/title cue on first mention in each scene "
                "(example shape: '모레노 CTO', '밀러 투자 파트너').\n\n"
            )
        if any(k in feedback_corpus for k in ("초반", "따라가기 힘들", "맥락", "인물 설명 없이", "누군지")):
            prompt += (
                "For opening scenes, add one brief orientation line clarifying who is present and why this exchange matters.\n\n"
            )
        prompt += (
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
            temperature=float(self.runtime_policy.get("distiller_temperature", 0.3) or 0.3),
            max_tokens=int(self.runtime_policy.get("distiller_max_tokens", 4000) or 4000),
        )

        scenes_data = self._parse_json_array(result)

        if not scenes_data:
            logger.warning("LLM distillation returned no scenes; falling back to chunking")
            return self._fallback_chunk(interactions, beats, target_scenes)

        # Convert to DistilledScene objects
        scenes: list[DistilledScene] = []
        available_turns = [
            self._coerce_turn_number(ix.get("turn"), default=0)
            for ix in interactions
        ]
        available_turns = sorted(turn for turn in available_turns if turn > 0)
        for i, sd in enumerate(scenes_data):
            payload = self._normalize_scene_payload(sd)
            turn_start, turn_end = self._coerce_scene_turn_range(
                payload,
                available_turns=available_turns,
                scene_index=i,
                total_scenes=len(scenes_data),
            )
            scene = DistilledScene(
                scene_number=i + 1,
                title=payload.get("title") or f"Scene {i + 1}",
                turn_range=(turn_start, turn_end),
                location=payload.get("location") or ep_location,
                characters_present=payload.get("characters", []),
                key_dialogue=payload.get("key_dialogue", []),
                key_actions=payload.get("key_actions", []),
                discoveries=payload.get("discoveries", []),
                emotional_arc=payload.get("emotional_arc", ""),
                beat_references=payload.get("beat_refs", []),
                narrative_summary=payload.get("summary", ""),
                pacing=payload.get("pacing", "building"),
                raw_turn_count=max(1, turn_end - turn_start + 1),
            )
            self._apply_scene_readability_guards(scene, canonical_speakers)
            scenes.append(scene)

        # Keep scene order stable by timeline and re-number deterministically.
        scenes.sort(key=lambda s: (s.turn_range[0], s.turn_range[1], s.scene_number))

        merge_budget = self._scene_merge_budget(target_scenes)
        if merge_budget > 0:
            scenes = self._merge_low_signal_adjacent_scenes(scenes, merge_budget)

        for idx, sc in enumerate(scenes, start=1):
            sc.scene_number = idx

        # Deterministic beat-reference reinforcement from actual director clue events.
        # This prevents LLM scene summaries from "forgetting" clue IDs present in log.
        turn_to_clues: dict[int, list[str]] = {}
        for ix in interactions:
            if ix.get("action_type") != "director_event":
                continue
            md = ix.get("metadata", {}) or {}
            clue_id = str(md.get("clue_id", "")).strip()
            if not clue_id:
                continue
            turn = self._coerce_turn_number(ix.get("turn", 0), default=0)
            if turn <= 0:
                continue
            turn_to_clues.setdefault(turn, []).append(clue_id)

        for s in scenes:
            start, end = s.turn_range
            forced: list[str] = []
            for t in range(start, end + 1):
                forced.extend(turn_to_clues.get(t, []))
            if forced:
                s.beat_references = self._dedupe_preserve_order(
                    list(s.beat_references) + forced
                )

        # Validate beat coverage
        covered_beats = set()
        for s in scenes:
            covered_beats.update(s.beat_references)
        required = {b["id"] for b in beats}
        missing = required - covered_beats
        if missing:
            logger.warning("Beats not covered in distilled scenes: %s", missing)
            # Assign missing beats to most semantically relevant scene.
            for beat_id in sorted(missing):
                beat_text = next(
                    (b.get("content", "") for b in beats if b.get("id") == beat_id),
                    "",
                )
                best = max(
                    scenes,
                    key=lambda s: self._token_overlap_score(
                        beat_text,
                        " ".join(s.discoveries)
                        + " "
                        + s.narrative_summary
                        + " "
                        + " ".join(a.get("line", "") for a in s.key_dialogue if isinstance(a, dict))
                        + " "
                        + " ".join(s.key_actions),
                    ),
                )
                best.beat_references = self._dedupe_preserve_order(
                    list(best.beat_references) + [beat_id]
                )

        return self.apply_scene_guards(scenes)

    def apply_scene_guards(
        self,
        scenes: list[DistilledScene],
        canonical_speakers: Optional[dict[str, str]] = None,
    ) -> list[DistilledScene]:
        guarded = list(scenes or [])
        canonical = canonical_speakers or {}
        for scene in guarded:
            self._apply_scene_readability_guards(scene, canonical)
        self._clarify_adjacent_character_entries(guarded)
        return guarded

    def _apply_scene_readability_guards(
        self,
        scene: DistilledScene,
        canonical_speakers: Optional[dict[str, str]] = None,
    ) -> None:
        canonical = canonical_speakers or {}
        self._sanitize_scene_dialogue(scene)
        self._compress_expository_dialogue(scene)
        scene.narrative_summary = self._tighten_narrative_summary(scene.narrative_summary)
        scene.narrative_summary = self._rebalance_narrative_summary(scene)
        scene.narrative_summary = self._soften_technical_summary(scene)
        scene.narrative_summary = self._simplify_summary_wording(scene.narrative_summary)
        scene.narrative_summary = self._compress_repeated_core_concerns(scene.narrative_summary)
        scene.narrative_summary = self._rebalance_summary_sentence_rhythm(scene.narrative_summary)
        scene.narrative_summary = self._trim_summary_leading_connectors(scene.narrative_summary)
        scene.narrative_summary = self._enforce_summary_sentence_word_cap(scene.narrative_summary)
        scene.narrative_summary = self._trim_summary_leading_connectors(scene.narrative_summary)
        scene.narrative_summary = self._tighten_narrative_summary(scene.narrative_summary)
        scene.characters_present = self._canonicalize_name_list(
            scene.characters_present,
            canonical,
        )
        for row in scene.key_dialogue:
            if not isinstance(row, dict):
                continue
            row["speaker"] = self._canonicalize_name(
                str(row.get("speaker", "")).strip(),
                canonical,
            )
        scene.key_actions = self._compress_core_concern_lines(scene.key_actions, limit=8)
        scene.discoveries = self._compress_core_concern_lines(scene.discoveries, limit=6)
        scene.characters_present = self._normalize_scene_character_labels(scene.characters_present)
        scene.narrative_summary = self._normalize_character_mentions(
            scene.narrative_summary,
            scene.characters_present,
        )
        scene.key_actions = [
            self._normalize_character_mentions(line, scene.characters_present)
            for line in scene.key_actions
        ]
        scene.discoveries = [
            self._normalize_character_mentions(line, scene.characters_present)
            for line in scene.discoveries
        ]

    def _reader_prefers_compact_beats(self) -> bool:
        corpus = self._reader_feedback_corpus()
        return any(
            k in corpus for k in (
                "기술", "용어", "약어", "약자", "전문", "jargon", "acronym",
                "반복", "중복", "리스트", "목록", "나열", "정보 전달",
            )
        )

    def _reader_feedback_corpus(self) -> str:
        if not self.reader_feedback:
            return ""
        return " ".join(
            str(x)
            for key in ("what_felt_boring_or_hard", "style_tips", "reader_comment")
            for x in (
                self.reader_feedback.get(key, [])
                if isinstance(self.reader_feedback.get(key, []), list)
                else [self.reader_feedback.get(key, "")]
            )
            if str(x).strip()
        ).lower()

    def _reader_reports_stalled_progression(self) -> bool:
        corpus = self._reader_feedback_corpus()
        return any(
            token in corpus for token in (
                "멈춘 이유",
                "멈춘",
                "멈춤",
                "정체",
                "제자리",
                "안 나가",
                "진행이 안",
                "흐름이 끊",
            )
        )

    def _reader_prefers_faster_progression(self) -> bool:
        corpus = self._reader_feedback_corpus()
        return self._reader_reports_stalled_progression() or any(
            token in corpus for token in (
                "전개가 느려",
                "느려서 집중",
                "집중력을 잃",
                "늘어지",
                "템포가 느려",
                "속도감이 떨어",
            )
        )

    def _reader_needs_contextual_summaries(self) -> bool:
        corpus = self._reader_feedback_corpus()
        return any(
            token in corpus for token in (
                "간결한 문장",
                "문맥 파악",
                "맥락 파악",
                "따라가기 힘들",
                "문맥이 어려",
            )
        )

    def _reader_prefers_stronger_scene_compaction(self) -> bool:
        corpus = self._reader_feedback_corpus()
        return self._reader_reports_stalled_progression() or self._feedback_scene_compaction_target() <= 85 or any(
            token in corpus for token in (
                "반복되는 표현",
                "비슷한 상황",
                "비슷한 상황과 묘사",
                "묘사가 반복",
                "문장이 너무 길",
                "길고 복잡",
                "이해하기 어려",
                "지루",
            )
        )

    def _feedback_scene_compaction_target(self, default: int = 100) -> int:
        constraints = self.reader_feedback.get("style_constraints", {}) if self.reader_feedback else {}
        if not isinstance(constraints, dict):
            return default
        raw = constraints.get("scene_compaction_ratio_target", default)
        try:
            ratio = int(raw)
        except (TypeError, ValueError):
            ratio = default
        return max(60, min(100, ratio))

    def _scene_merge_budget(self, target_scenes: int) -> int:
        budget = 0
        if self._reader_reports_stalled_progression():
            budget += 1
        if self._reader_prefers_faster_progression():
            budget += 1
        if self._reader_needs_contextual_summaries():
            budget += 1
        if self._reader_prefers_stronger_scene_compaction():
            budget += 1
        return max(0, min(budget, max(0, target_scenes - 3)))

    def _merge_low_signal_adjacent_scenes(
        self,
        scenes: list[DistilledScene],
        merge_budget: int,
    ) -> list[DistilledScene]:
        merged = list(scenes)
        merges_left = max(0, merge_budget)

        while merges_left > 0 and len(merged) > 3:
            best_idx = -1
            best_score = 0
            for idx in range(len(merged) - 1):
                score = self._adjacent_scene_merge_score(merged[idx], merged[idx + 1])
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if self._reader_reports_stalled_progression():
                merge_threshold = 2
            else:
                merge_threshold = 3 if self._reader_prefers_stronger_scene_compaction() else 4
            if best_idx < 0 or best_score < merge_threshold:
                break
            merged[best_idx: best_idx + 2] = [
                self._merge_scene_pair(merged[best_idx], merged[best_idx + 1])
            ]
            merges_left -= 1

        return merged

    def _adjacent_scene_merge_score(self, left: DistilledScene, right: DistilledScene) -> int:
        score = 0
        tension_peak = max(self._scene_tension_score(left), self._scene_tension_score(right))
        same_location = (
            self._norm_name_key(left.location)
            and self._norm_name_key(left.location) == self._norm_name_key(right.location)
        )
        shared_chars = set(self._canonicalize_name_list(left.characters_present, {})) & set(
            self._canonicalize_name_list(right.characters_present, {})
        )
        if same_location:
            score += 3
        elif not shared_chars:
            return 0
        else:
            score += 1
        if shared_chars:
            score += 1
        if left.raw_turn_count <= 4 or right.raw_turn_count <= 4:
            score += 1
        if (left.raw_turn_count + right.raw_turn_count) <= 10:
            score += 1
        if len(left.discoveries) + len(right.discoveries) <= 2:
            score += 1
        if len(left.key_dialogue) + len(right.key_dialogue) <= 3:
            score += 1
        if len((left.narrative_summary or "").strip()) <= 90 or len((right.narrative_summary or "").strip()) <= 90:
            score += 1
        if self._scene_core_concern_signature(left) and (
            self._scene_core_concern_signature(left) == self._scene_core_concern_signature(right)
        ):
            score += 2
        if self._token_overlap_score(left.narrative_summary, right.narrative_summary) >= 0.35:
            score += 2
        if self._summary_is_mood_heavy(left.narrative_summary) or self._summary_is_mood_heavy(right.narrative_summary):
            score += 1
        if left.pacing == right.pacing:
            score += 1
        if tension_peak >= 6:
            score -= 3
        elif tension_peak >= 4:
            score -= 1
        elif self._scene_tension_score(left) + self._scene_tension_score(right) <= 2:
            score += 1
        return score

    def _summary_is_mood_heavy(self, summary: str) -> bool:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(summary or "").strip())
            if s.strip()
        ]
        if not sentences:
            return False
        mood_only = sum(1 for sent in sentences if self._is_mood_fragment_sentence(sent))
        return mood_only >= max(1, len(sentences) - 1)

    def _scene_tension_score(self, scene: DistilledScene) -> int:
        text = " ".join(
            [
                str(scene.narrative_summary or ""),
                " ".join(str(x) for x in (scene.discoveries or []) if str(x).strip()),
                " ".join(str(x) for x in (scene.key_actions or []) if str(x).strip()),
                " ".join(str((row or {}).get("line", "")) for row in (scene.key_dialogue or []) if isinstance(row, dict)),
                str(scene.emotional_arc or ""),
            ]
        )
        low = text.lower()
        score = 0
        score += min(2, len(scene.discoveries or []))
        score += min(1, len(scene.key_dialogue or []))
        if re.search(r"(드러났|밝혀졌|확인됐|폭로|반전|결정|선택|거절|수락|중단|경고|위험|충돌|대치)", low):
            score += 2
        if re.search(r"(긴장|압박|불안|초조|경계|결심|분노|공포|당황|침묵|정적|날카롭)", low):
            score += 1
        if re.search(r"(숨|손끝|손바닥|시선|턱선|몸이 먼저|멈칫|굳었|떨렸|멎은 듯)", low):
            score += 1
        return min(score, 7)

    def _trim_summary_leading_connectors(self, summary: str) -> str:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(summary or "").strip())
            if s.strip()
        ]
        if not sentences:
            return ""

        rebuilt: list[str] = []
        seen_fp: set[str] = set()
        for sent in sentences:
            cleaned = re.sub(
                r"^(그리고|그러자|다만|또한|한편|이어서|그 순간)\s+",
                "",
                sent,
            ).strip()
            if not cleaned:
                continue
            normalized = self._ensure_summary_sentence(cleaned)
            fp = self._dialogue_fingerprint(normalized)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            rebuilt.append(normalized)
        max_sentences = 2 if self._reader_prefers_stronger_scene_compaction() else 3
        return " ".join(rebuilt[:max_sentences]).strip()

    def _merge_scene_pair(self, left: DistilledScene, right: DistilledScene) -> DistilledScene:
        pacing = right.pacing if right.pacing in {"climax", "resolution"} else left.pacing
        if left.pacing == "climax":
            pacing = left.pacing
        location = left.location if self._norm_name_key(left.location) == self._norm_name_key(right.location) else (
            right.location or left.location
        )
        emotional_parts = [p for p in (left.emotional_arc, right.emotional_arc) if str(p).strip()]
        summary = self._merge_scene_summaries(left.narrative_summary, right.narrative_summary)
        return DistilledScene(
            scene_number=left.scene_number,
            title=left.title,
            turn_range=(left.turn_range[0], right.turn_range[1]),
            location=location,
            characters_present=self._dedupe_preserve_order(left.characters_present + right.characters_present),
            key_dialogue=(left.key_dialogue + right.key_dialogue)[:4],
            key_actions=self._dedupe_semantic_lines(left.key_actions + right.key_actions, limit=8),
            discoveries=self._dedupe_semantic_lines(left.discoveries + right.discoveries, limit=6),
            emotional_arc=" -> ".join(emotional_parts[:2]),
            beat_references=self._dedupe_preserve_order(left.beat_references + right.beat_references),
            narrative_summary=summary,
            pacing=pacing,
            raw_turn_count=left.raw_turn_count + right.raw_turn_count,
        )

    def _merge_scene_summaries(self, left: str, right: str) -> str:
        parts: list[str] = []
        seen_fp: set[str] = set()
        for raw in (left, right):
            for sentence in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(raw or "").strip()):
                sent = sentence.strip()
                if not sent:
                    continue
                fp = self._dialogue_fingerprint(sent)
                if not fp or fp in seen_fp:
                    continue
                seen_fp.add(fp)
                parts.append(sent)
        max_sentences = 2 if self._reader_prefers_stronger_scene_compaction() else 3
        return self._compress_repeated_core_concerns(" ".join(parts[:max_sentences]).strip())

    def _tighten_narrative_summary(self, summary: str) -> str:
        sentences: list[str] = []
        seen_fp: set[str] = set()
        for raw in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(summary or "").strip()):
            sent = raw.strip()
            if not sent:
                continue
            fp = self._dialogue_fingerprint(sent)
            if not fp or fp in seen_fp:
                continue
            seen_fp.add(fp)
            sentences.append(sent)
        max_sentences = 2 if self._reader_prefers_stronger_scene_compaction() else 3
        return self._compress_repeated_core_concerns(" ".join(sentences[:max_sentences]).strip())

    def _rebalance_narrative_summary(self, scene: DistilledScene) -> str:
        """
        Prefer action/reaction summary over repeated mood fragments.
        Keep at most one atmosphere beat when it carries actual pressure.
        """
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(scene.narrative_summary or "").strip())
            if s.strip()
        ]
        if not raw_sentences:
            return self._summary_replacement_sentence(scene)

        keep: list[str] = []
        mood_kept = 0
        for sent in raw_sentences:
            if not self._is_mood_fragment_sentence(sent):
                normalized = self._ensure_summary_sentence(sent)
                if normalized and normalized not in keep:
                    keep.append(normalized)
                continue
            if mood_kept == 0 and self._scene_can_keep_mood_fragment(scene):
                normalized = self._blend_mood_fragment_with_consequence(sent, scene)
                if normalized and normalized not in keep:
                    keep.append(normalized)
                mood_kept += 1
                continue
            replacement = self._summary_replacement_sentence(scene)
            if replacement and replacement not in keep:
                keep.append(replacement)

        if not keep:
            replacement = self._summary_replacement_sentence(scene)
            if replacement:
                keep.append(replacement)

        if keep and not any(self._summary_has_action_or_decision(sent) for sent in keep):
            replacement = self._summary_replacement_sentence(scene)
            if replacement:
                keep[0] = replacement

        max_sentences = 2 if self._reader_prefers_stronger_scene_compaction() else 3
        return self._compress_repeated_core_concerns(" ".join(keep[:max_sentences]).strip())

    def _blend_mood_fragment_with_consequence(self, sentence: str, scene: DistilledScene) -> str:
        fragment = re.sub(r"\s+", " ", str(sentence or "")).strip()
        if not fragment:
            return self._summary_replacement_sentence(scene)
        if self._summary_has_action_or_decision(fragment):
            return self._ensure_summary_sentence(fragment)
        replacement = self._summary_replacement_sentence(scene)
        if not replacement:
            return self._ensure_summary_sentence(fragment)
        base = re.sub(r"[.!?…]+$", "", fragment).strip()
        follow = re.sub(r"^[\"“”'‘’\s]+", "", replacement).strip()
        follow = re.sub(r"[.!?…]+$", "", follow).strip()
        if not base or not follow:
            return self._ensure_summary_sentence(fragment or replacement)
        if self._dialogue_fingerprint(base) == self._dialogue_fingerprint(follow):
            return self._ensure_summary_sentence(base)
        return self._ensure_summary_sentence(f"{base}, 그래서 {follow}")

    def _scene_can_keep_mood_fragment(self, scene: DistilledScene) -> bool:
        if scene.pacing in {"opening", "climax"}:
            return True
        if not scene.key_actions and not scene.discoveries:
            return True
        return False

    @staticmethod
    def _ensure_summary_sentence(text: str) -> str:
        sent = re.sub(r"\s+", " ", str(text or "")).strip()
        if not sent:
            return ""
        if re.search(r"[.!?…]$", sent):
            return sent
        return sent + "."

    @staticmethod
    def _summary_has_action_or_decision(sentence: str) -> bool:
        return bool(re.search(
            r"(건넸|밀었|열었|접었|돌렸|움직였|받았|붙잡|꺼냈|멈추|확인|드러났|알아차렸|결정|선택|반응|질문|대답|응답|설득|거절|숨을 골랐|고개를 들었)",
            str(sentence or ""),
        ))

    def _is_mood_fragment_sentence(self, sentence: str) -> bool:
        sent = re.sub(r"\s+", " ", str(sentence or "")).strip()
        if not sent:
            return False
        if self._summary_has_action_or_decision(sent):
            return False
        mood_hits = len(re.findall(
            r"(정적|침묵|공기|소음|복도|불빛|시선|손|표정|숨|기계음|발소리|거리|문|그 말이 공중에 남)",
            sent,
        ))
        if mood_hits == 0:
            return False
        token_count = len(re.findall(r"[0-9A-Za-z가-힣]+", sent))
        return token_count <= 9

    def _summary_replacement_sentence(self, scene: DistilledScene) -> str:
        for action in scene.key_actions or []:
            cleaned = self._clean_action_text(action)
            if cleaned:
                return self._ensure_summary_sentence(cleaned)
        for item in scene.discoveries or []:
            cleaned = re.sub(r"\s+", " ", str(item or "")).strip()
            if not cleaned:
                continue
            if re.search(r"(드러났|확인됐|밝혀졌|알아냈|감지됐|포착됐)", cleaned):
                return self._ensure_summary_sentence(cleaned)
            subject = scene.characters_present[0] if scene.characters_present else "인물들은"
            return self._ensure_summary_sentence(f"{subject}은 {cleaned}")
        emotional = re.sub(r"\s+", " ", str(scene.emotional_arc or "")).strip()
        if emotional:
            subject = scene.characters_present[0] if scene.characters_present else "인물들은"
            return self._ensure_summary_sentence(f"{subject}의 감정선은 {emotional}으로 기울었다")
        return self._ensure_summary_sentence(scene.narrative_summary)

    def _summary_sentence_word_cap(self, default: int = 18) -> int:
        constraints = self.reader_feedback.get("style_constraints", {}) if self.reader_feedback else {}
        if not isinstance(constraints, dict):
            return default
        raw = constraints.get("scene_summary_sentence_words_max", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(10, min(20, cap))

    def _force_reaction_after_jargon(self) -> bool:
        constraints = self.reader_feedback.get("style_constraints", {}) if self.reader_feedback else {}
        if not isinstance(constraints, dict):
            return False
        raw = constraints.get("force_reaction_after_jargon", 0)
        try:
            enabled = int(raw)
        except (TypeError, ValueError):
            enabled = 0
        return enabled >= 1

    def _soften_technical_summary(self, scene: DistilledScene) -> str:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(scene.narrative_summary or "").strip())
            if s.strip()
        ]
        if not sentences:
            return scene.narrative_summary

        rebuilt: list[str] = []
        added_reaction = False
        explanation_added = False
        for sent in sentences:
            normalized = self._ensure_summary_sentence(sent)
            if not self._summary_is_jargon_heavy(sent) or self._summary_has_reaction_or_sensory(sent):
                rebuilt.append(normalized)
                continue

            rebuilt.append(normalized)

            if self._summary_plain_buffer_enabled() and not explanation_added:
                preface = self._summary_plain_preface(sent)
                if preface:
                    rebuilt.append(preface)
                    explanation_added = True

            if self._force_reaction_after_jargon() or not added_reaction:
                reaction = self._summary_reaction_tail(scene)
                if reaction:
                    rebuilt.append(reaction)
                    added_reaction = True

        max_sentences = 3 if explanation_added else (2 if self._reader_prefers_stronger_scene_compaction() else 3)
        return " ".join(rebuilt[:max_sentences]).strip()

    def _summary_plain_buffer_enabled(self) -> bool:
        constraints = self.reader_feedback.get("style_constraints", {}) if self.reader_feedback else {}
        if not isinstance(constraints, dict):
            return True
        raw = constraints.get("jargon_buffer_sentences", 1)
        try:
            enabled = int(raw)
        except (TypeError, ValueError):
            enabled = 1
        return enabled >= 1

    @staticmethod
    def _simplify_summary_wording(summary: str) -> str:
        text = re.sub(r"\s+", " ", str(summary or "")).strip()
        if not text:
            return ""
        replacements = (
            ("요지는", "핵심은"),
            ("여파로", "그래서"),
            ("기울었다", "움직였다"),
            ("감정선은", "마음은"),
            ("드러났다는 뜻이었다", "드러난 셈이었다"),
            ("계산의 결이 흐트러질 수 있다는 뜻이었다", "계산이 흔들릴 수 있다는 말이었다"),
            ("수치가 조금씩 밀리고 있다는 뜻이었다", "수치가 조금씩 어긋난다는 말이었다"),
        )
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def _summary_plain_preface(self, sentence: str) -> str:
        cue = self._plain_term_cue(sentence)
        if cue:
            plain = re.sub(r"^즉\s*", "쉽게 말하면 ", cue).strip()
            metaphor = self._plain_term_metaphor(sentence)
            if metaphor and self._summary_easy_metaphor_enabled():
                plain = f"{plain}, {metaphor}"
            return self._ensure_summary_sentence(plain)
        return self._ensure_summary_sentence("쉽게 말하면 지금 바로 확인해야 할 문제가 드러난 셈이었다")

    def _summary_easy_metaphor_enabled(self) -> bool:
        constraints = self.reader_feedback.get("style_constraints", {}) if self.reader_feedback else {}
        if not isinstance(constraints, dict):
            return True
        raw = constraints.get("summary_easy_metaphor_once", 1)
        try:
            enabled = int(raw)
        except (TypeError, ValueError):
            enabled = 1
        return enabled >= 1

    @staticmethod
    def _summary_is_jargon_heavy(sentence: str) -> bool:
        raw = str(sentence or "")
        if not raw.strip():
            return False
        low = raw.lower()
        hits = len(re.findall(r"\b[A-Z]{2,8}(?:-\d+)?\b", raw))
        english_tokens = [
            token
            for token in re.findall(r"\b[a-z]{4,}(?:-\d+)?\b", low)
            if token not in {"there", "where", "which", "while", "about", "after", "before", "their"}
        ]
        if english_tokens:
            hits += min(2, len(english_tokens))
        for token in (
            "latency", "coherence", "drift", "protocol", "qpu",
            "지연", "결맞음", "드리프트", "편차", "프로토콜", "양자", "보정", "오차",
        ):
            if token in low:
                hits += 1
        return hits >= 1

    @staticmethod
    def _summary_has_reaction_or_sensory(sentence: str) -> bool:
        return bool(re.search(
            r"(손끝|손바닥|목 안|숨|호흡|시선|눈빛|귀|귀에|차갑|뜨겁|굳었|멈칫|흔들|고개를 들|답을 잇지 못했|몸이 먼저|표정이)",
            str(sentence or ""),
        ))

    @staticmethod
    def _plain_term_cue(sentence: str) -> str:
        low = str(sentence or "").lower()
        if "latency" in low or "지연" in low:
            return "즉 반응이 한 박자 늦는다는 뜻이었다"
        if "coherence" in low or "결맞음" in low:
            return "즉 계산의 결이 흐트러질 수 있다는 뜻이었다"
        if "drift" in low or "드리프트" in low or "편차" in low:
            return "즉 수치가 조금씩 밀리고 있다는 뜻이었다"
        if "qpu" in low or "양자 처리 칩" in low:
            return "장비의 핵심 칩 쪽 문제라는 뜻이었다"
        if "protocol" in low or "프로토콜" in low:
            return "현장 절차를 다시 밟아야 한다는 뜻이었다"
        if "보정" in low or "오차" in low:
            return "숫자를 다시 맞춰야 한다는 뜻이었다"
        return ""

    @staticmethod
    def _plain_term_metaphor(sentence: str) -> str:
        low = str(sentence or "").lower()
        if "latency" in low or "지연" in low:
            return "답이 반 박자 늦게 튀는 셈이었다"
        if "coherence" in low or "결맞음" in low:
            return "실 한 올이 풀리듯 계산의 결이 흐트러질 수 있었다"
        if "drift" in low or "드리프트" in low or "편차" in low:
            return "바늘이 조금씩 옆으로 밀리는 셈이었다"
        if "qpu" in low or "양자 처리 칩" in low:
            return "엔진칸이 흔들리면 장비 전체가 덜컹이는 것과 비슷했다"
        if "protocol" in low or "프로토콜" in low:
            return "정해 둔 비상 계단을 다시 밟는 셈이었다"
        if "보정" in low or "오차" in low:
            return "삐뚤어진 저울 눈금을 다시 맞추는 셈이었다"
        return ""

    def _summary_reaction_tail(self, scene: DistilledScene) -> str:
        subject = scene.characters_present[0] if scene.characters_present else "인물들"
        emotional = str(scene.emotional_arc or "")
        if re.search(r"(불안|긴장|초조|압박)", emotional):
            return self._ensure_summary_sentence(f"{subject}의 말끝이 잠깐 무거워졌다")
        if re.search(r"(안도|진정|안정)", emotional):
            return self._ensure_summary_sentence(f"{subject}은 그제야 어깨 힘을 조금 풀었다")
        if re.search(r"(분노|짜증|격앙)", emotional):
            return self._ensure_summary_sentence(f"{subject}의 턱선이 눈에 띄게 굳었다")
        if re.search(r"(혼란|당황|망설임)", emotional):
            return self._ensure_summary_sentence(f"{subject}은 잠깐 답을 잇지 못했다")
        return self._ensure_summary_sentence(f"{subject}은 주변의 반응을 먼저 살폈다")

    def _rebalance_summary_sentence_rhythm(self, summary: str) -> str:
        sentences = [
            self._ensure_summary_sentence(s)
            for s in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(summary or "").strip())
            if s.strip()
        ]
        if len(sentences) < 3:
            return summary

        rebuilt = list(sentences)
        idx = 1
        while idx < len(rebuilt):
            prev = rebuilt[idx - 1]
            current = rebuilt[idx]
            prev_words = self._summary_word_count(prev)
            current_words = self._summary_word_count(current)
            if abs(prev_words - current_words) <= 2:
                merged = self._merge_summary_pair(prev, current)
                if merged:
                    rebuilt[idx - 1:idx + 1] = [merged]
                    continue
            idx += 1

        max_sentences = 3 if len(rebuilt) >= 3 else len(rebuilt)
        return " ".join(rebuilt[:max_sentences]).strip()

    def _enforce_summary_sentence_word_cap(self, summary: str) -> str:
        cap = self._summary_sentence_word_cap(default=18)
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(summary or "").strip())
            if s.strip()
        ]
        if not sentences:
            return summary

        rebuilt: list[str] = []
        for sent in sentences:
            rebuilt.extend(self._split_summary_sentence_by_word_cap(sent, max_words=cap))
        max_sentences = 3 if self._reader_prefers_stronger_scene_compaction() else 4
        capped = " ".join(self._ensure_summary_sentence(s) for s in rebuilt[:max_sentences] if s.strip()).strip()
        return self._compress_repeated_core_concerns(capped)

    @staticmethod
    def _split_summary_sentence_by_word_cap(sentence: str, max_words: int = 15) -> list[str]:
        sent = re.sub(r"\s+", " ", str(sentence or "")).strip()
        if not sent:
            return []
        comma_heavy = (sent.count(",") + sent.count("，") + sent.count(";")) >= 2
        connective_heavy = len(re.findall(r"(그리고|그러나|하지만|다만|그래서|그러자|한편|또한)", sent)) >= 2
        if len(re.findall(r"[0-9A-Za-z가-힣]+", sent)) <= max_words and not comma_heavy and not connective_heavy:
            return [re.sub(r"[.!?…]+$", "", sent).strip()]

        clauses = re.split(
            r"(?<=[,，;])\s+|(?<=다)\s+(?=그리고|그러나|하지만|다만|또는|또|한편|그래서|그러자|그때)",
            sent,
        )
        clauses = [re.sub(r"[.!?…]+$", "", clause).strip() for clause in clauses if clause.strip()]
        if len(clauses) <= 1:
            words = sent.split()
            out: list[str] = []
            for idx in range(0, len(words), max_words):
                chunk = " ".join(words[idx:idx + max_words]).strip()
                if chunk:
                    out.append(re.sub(r"[.!?…]+$", "", chunk).strip())
            return out or [re.sub(r"[.!?…]+$", "", sent).strip()]

        out: list[str] = []
        buf: list[str] = []
        buf_words = 0
        for clause in clauses:
            clause_words = len(re.findall(r"[0-9A-Za-z가-힣]+", clause))
            if buf and (buf_words + clause_words > max_words):
                out.append(" ".join(buf).strip())
                buf = []
                buf_words = 0
            buf.append(clause)
            buf_words += clause_words
        if buf:
            out.append(" ".join(buf).strip())
        return out or [re.sub(r"[.!?…]+$", "", sent).strip()]

    @staticmethod
    def _summary_word_count(sentence: str) -> int:
        return len(re.findall(r"[0-9A-Za-z가-힣]+", str(sentence or "")))

    def _merge_summary_pair(self, left: str, right: str) -> str:
        left_base = re.sub(r"[.!?…]+$", "", str(left or "").strip())
        right_base = re.sub(r"[.!?…]+$", "", str(right or "").strip())
        if not left_base or not right_base:
            return ""
        if self._dialogue_fingerprint(left_base) == self._dialogue_fingerprint(right_base):
            return self._ensure_summary_sentence(left_base)
        if self._summary_has_action_or_decision(left_base) and not self._summary_has_action_or_decision(right_base):
            return self._ensure_summary_sentence(f"{left_base}, 그래서 {right_base}")
        if not self._summary_has_action_or_decision(left_base) and self._summary_has_action_or_decision(right_base):
            return self._ensure_summary_sentence(f"{left_base}, 그래서 {right_base}")
        if self._summary_word_count(left_base) <= 8 and self._summary_word_count(right_base) <= 8:
            return self._ensure_summary_sentence(f"{left_base}, 그래서 {right_base}")
        return ""

    @staticmethod
    def _compress_beat_content(text: str, max_chars: int = 180) -> str:
        """
        Keep beat fidelity while preventing clue payloads from over-dominating
        the distillation prompt with repeated technical listing.
        """
        raw = re.sub(r"\s+", " ", str(text or "")).strip()
        if not raw:
            return ""
        parts = re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", raw)
        kept: list[str] = []
        for part in parts:
            p = part.strip()
            if not p:
                continue
            if kept and p.lower() == kept[-1].lower():
                continue
            kept.append(p)
            if len(" ".join(kept)) >= max_chars:
                break
        compact = " ".join(kept).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max(0, max_chars - 1)].rstrip() + "…"

    def _sanitize_scene_dialogue(self, scene: DistilledScene) -> None:
        """
        Normalize distilled dialogue so action narration is not treated as spoken quotes.
        Moves non-spoken lines into key_actions.
        """
        cleaned_dialogue: list[dict] = []
        extra_actions: list[str] = []
        seen_lines: set[str] = set()

        for row in scene.key_dialogue or []:
            if not isinstance(row, dict):
                continue
            speaker = str(row.get("speaker", "")).strip() or "Unknown"
            line = str(row.get("line", "")).strip()
            if not line:
                continue

            spoken = self._extract_spoken_dialogue(line)
            if spoken:
                speaker_fp = self._dialogue_fingerprint(speaker)
                line_fp = self._dialogue_fingerprint(spoken)
                fp = f"{speaker_fp}::{line_fp}" if line_fp else ""
                if fp and fp in seen_lines:
                    continue
                if fp:
                    seen_lines.add(fp)
                cleaned_dialogue.append({"speaker": speaker, "line": spoken})
                continue

            # Treat narration-like entries as actions, not spoken lines.
            action = self._clean_action_text(line)
            if action:
                extra_actions.append(f"{speaker}: {action}")

        scene.key_dialogue = cleaned_dialogue[:4]
        if extra_actions:
            scene.key_actions = self._dedupe_preserve_order(list(scene.key_actions) + extra_actions)

    def _compress_expository_dialogue(self, scene: DistilledScene) -> None:
        """
        Trim info-dump style quotes and preserve impact via short action beats.
        """
        compact: list[dict] = []
        extra_actions: list[str] = []
        explain_pattern = re.compile(r"(왜냐하면|즉|다시 말해|정리하면|요약하면|핵심은|결론은|설명하자면)")
        for row in scene.key_dialogue or []:
            if not isinstance(row, dict):
                continue
            speaker = str(row.get("speaker", "")).strip() or "Unknown"
            line = re.sub(r"\s+", " ", str(row.get("line", "") or "")).strip()
            if not line:
                continue
            if len(line) > 90 and explain_pattern.search(line):
                cut = re.split(r"[,;]|(?:\s+그리고\s+)|(?:\s+하지만\s+)", line, maxsplit=1)[0].strip()
                if cut and cut != line:
                    line = cut.rstrip(". ") + "…"
                extra_actions.append(f"{speaker}는 말을 짧게 정리하고 반응을 살폈다.")
            compact.append({"speaker": speaker, "line": line})
        scene.key_dialogue = compact[:4]
        if extra_actions:
            scene.key_actions = self._dedupe_preserve_order(list(scene.key_actions) + extra_actions)

    def _compress_repeated_core_concerns(self, summary: str) -> str:
        sentences = [
            self._ensure_summary_sentence(s)
            for s in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", str(summary or "").strip())
            if s.strip()
        ]
        if len(sentences) < 2:
            return summary

        kept: list[str] = []
        concern_index: dict[str, int] = {}
        for sentence in sentences:
            sig = self._core_concern_signature(sentence)
            if sig:
                prev_idx = concern_index.get(sig)
                if prev_idx is not None and self._is_redundant_core_concern(kept[prev_idx], sentence):
                    kept[prev_idx] = self._prefer_more_concrete_line(kept[prev_idx], sentence)
                    continue
            kept.append(sentence)
            if sig:
                concern_index[sig] = len(kept) - 1
        return " ".join(kept).strip()

    def _compress_core_concern_lines(self, values: list[str], limit: int) -> list[str]:
        compact = self._dedupe_semantic_lines(values, limit=max(1, limit * 2))
        kept: list[str] = []
        concern_index: dict[str, int] = {}
        for line in compact:
            sig = self._core_concern_signature(line)
            if sig:
                prev_idx = concern_index.get(sig)
                if prev_idx is not None and self._is_redundant_core_concern(kept[prev_idx], line):
                    kept[prev_idx] = self._prefer_more_concrete_line(kept[prev_idx], line)
                    continue
            kept.append(line)
            if sig:
                concern_index[sig] = len(kept) - 1
            if len(kept) >= max(1, limit):
                break
        return kept

    def _core_concern_signature(self, text: str) -> str:
        low = str(text or "").lower()
        parts: list[str] = []
        if re.search(r"(외부 지원|지원 구조|지원|후원|자원|장비|인력|예산|환경|resource|support|funding)", low):
            parts.append("support")
        if re.search(r"(실시간|real-time|latency|지연|보정|control loop|제어 루프|compensation)", low):
            parts.append("realtime")
        if re.search(r"(통제|통제권|누가 쥐|누가 결정|주도권|권한|authority|control)", low):
            parts.append("control")
        if re.search(r"(책임|책임질|법적|감시 한도|oversight|accountability|liability)", low):
            parts.append("responsibility")
        return "|".join(parts[:3])

    def _scene_core_concern_signature(self, scene: DistilledScene) -> str:
        return self._core_concern_signature(
            " ".join(
                [
                    str(scene.narrative_summary or ""),
                    " ".join(str(x) for x in (scene.discoveries or []) if str(x).strip()),
                    " ".join(str(x) for x in (scene.key_actions or []) if str(x).strip()),
                    str(scene.emotional_arc or ""),
                ]
            )
        )

    def _is_redundant_core_concern(self, left: str, right: str) -> bool:
        left_sig = self._core_concern_signature(left)
        right_sig = self._core_concern_signature(right)
        if not left_sig or left_sig != right_sig:
            return False
        left_tokens = self._line_fingerprint_set(left)
        right_tokens = self._line_fingerprint_set(right)
        if not left_tokens or not right_tokens:
            return False
        overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
        both_static = (
            not self._summary_has_action_or_decision(left)
            and not self._summary_has_action_or_decision(right)
        )
        return overlap >= 0.25 or both_static

    def _prefer_more_concrete_line(self, left: str, right: str) -> str:
        def score(line: str) -> tuple[int, int]:
            concrete = 0
            if self._summary_has_action_or_decision(line):
                concrete += 3
            if re.search(r"[0-9]", str(line or "")):
                concrete += 1
            if re.search(r"[A-Z]{2,}|[\"“”'‘’]", str(line or "")):
                concrete += 1
            return concrete, -self._summary_word_count(line)

        return right if score(right) > score(left) else left

    def _normalize_scene_character_labels(self, names: list[str]) -> list[str]:
        named_identity = self._preferred_named_identity(names)
        out: list[str] = []
        seen: set[str] = set()
        for raw in names or []:
            fixed = str(raw or "").strip()
            if not fixed:
                continue
            if self._looks_like_generic_suit_label(fixed):
                fixed = named_identity if named_identity and self._is_miller_identity(named_identity) else "이름 모를 수트 차림의 남자"
            fp = self._norm_name_key(fixed)
            if fp in seen:
                continue
            seen.add(fp)
            out.append(fixed)
        return out

    def _normalize_character_mentions(self, text: str, character_names: list[str]) -> str:
        raw = str(text or "")
        if not raw.strip():
            return raw
        named_identity = self._preferred_named_identity(character_names)
        replacement = named_identity if named_identity and self._is_miller_identity(named_identity) else "이름 모를 수트 차림의 남자"
        pattern = re.compile(
            r"(?:다크|짙은|남색|검은|회색)\s*수트(?:\s*차림)?(?:의)?\s*(?:남자|사내)"
            r"|수트(?:\s*차림)?(?:의)?\s*(?:남자|사내)"
            r"|dark[- ]suit(?:ed)?\s+(?:man|figure)"
            r"|man in (?:a )?dark suit",
            re.IGNORECASE,
        )
        return pattern.sub(replacement, raw)

    def _clarify_adjacent_character_entries(self, scenes: list[DistilledScene]) -> None:
        for idx, scene in enumerate(scenes):
            scene.characters_present = self._normalize_scene_character_labels(scene.characters_present)
            scene.narrative_summary = self._normalize_character_mentions(
                scene.narrative_summary,
                scene.characters_present,
            )
            scene.key_actions = [
                self._normalize_character_mentions(line, scene.characters_present)
                for line in scene.key_actions
            ]
            scene.discoveries = [
                self._normalize_character_mentions(line, scene.characters_present)
                for line in scene.discoveries
            ]
            if idx == 0:
                continue
            prev = scenes[idx - 1]
            if (
                self._scene_has_unnamed_suit_observer(prev)
                and self._scene_has_named_miller(scene)
            ):
                scene.narrative_summary = self._normalize_character_mentions(
                    scene.narrative_summary,
                    scene.characters_present,
                )

    def _preferred_named_identity(self, names: list[str]) -> str:
        for name in names or []:
            raw = str(name or "").strip()
            if raw and self._is_miller_identity(raw):
                return raw
        for name in names or []:
            raw = str(name or "").strip()
            if raw and not self._looks_like_generic_suit_label(raw):
                return raw
        return ""

    @staticmethod
    def _looks_like_generic_suit_label(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        return bool(re.search(
            r"(?:다크|짙은|남색|검은|회색)\s*수트|수트\s*차림|dark[- ]suit|suited",
            raw,
            re.IGNORECASE,
        )) and bool(re.search(r"(남자|사내|man|figure)", raw, re.IGNORECASE))

    @staticmethod
    def _is_miller_identity(text: str) -> bool:
        low = str(text or "").lower()
        return "miller" in low or "밀러" in low

    def _scene_has_named_miller(self, scene: DistilledScene) -> bool:
        return any(self._is_miller_identity(name) for name in (scene.characters_present or []))

    def _scene_has_unnamed_suit_observer(self, scene: DistilledScene) -> bool:
        text = " ".join(
            [
                scene.narrative_summary or "",
                " ".join(scene.key_actions or []),
                " ".join(scene.discoveries or []),
                " ".join(scene.characters_present or []),
            ]
        )
        return "이름 모를 수트 차림의 남자" in text or self._looks_like_generic_suit_label(text)

    @staticmethod
    def _dialogue_fingerprint(text: str) -> str:
        cleaned = re.sub(r"[^0-9a-z가-힣\s]", " ", str(text or "").lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _norm_name_key(text: str) -> str:
        return re.sub(r"[^0-9a-z가-힣]+", "", str(text or "").lower())

    def _build_canonical_speaker_map(self, interactions: list[dict]) -> dict[str, str]:
        """
        Build a conservative alias->canonical display-name map from raw interaction logs.
        This stabilizes name attribution when LLM distillation outputs mixed labels/titles.
        """
        key_to_names: dict[str, set[str]] = {}
        for ix in interactions:
            if str(ix.get("speaker_id", "")).strip() == "director":
                continue
            canonical = str(ix.get("speaker_name", "")).strip()
            if not canonical:
                continue
            raw_parts = [canonical]
            raw_parts.extend(
                token for token in re.split(r"[\s\-_/]+", canonical)
                if len(token.strip()) >= 2
            )
            if re.fullmatch(r"[가-힣]{3,4}", canonical):
                raw_parts.append(canonical[-2:])
            for part in raw_parts:
                key = self._norm_name_key(part)
                if not key:
                    continue
                key_to_names.setdefault(key, set()).add(canonical)
        return {
            key: next(iter(names))
            for key, names in key_to_names.items()
            if len(names) == 1
        }

    def _canonicalize_name(self, name: str, canonical_map: dict[str, str]) -> str:
        raw = str(name or "").strip()
        if not raw:
            return raw
        key = self._norm_name_key(raw)
        if key in canonical_map:
            return canonical_map[key]
        for cand_key, canonical in canonical_map.items():
            if len(cand_key) < 2:
                continue
            if key and (key in cand_key or cand_key in key):
                return canonical
        return raw

    def _canonicalize_name_list(self, names: list[str], canonical_map: dict[str, str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in names or []:
            fixed = self._canonicalize_name(str(raw or "").strip(), canonical_map)
            if not fixed:
                continue
            fp = self._norm_name_key(fixed)
            if fp in seen:
                continue
            seen.add(fp)
            out.append(fixed)
        return out

    @staticmethod
    def _extract_spoken_dialogue(text: str) -> str:
        """Extract spoken quote if present; otherwise return empty."""
        if not text:
            return ""
        # Strip obvious inner-thought spans.
        cleaned = re.sub(r"\[\[.*?\]\]|\[.*?\]", "", text).strip()

        # Capture quoted speech first.
        quoted = re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,})[\"“”'‘’]", cleaned)
        if quoted:
            return quoted[0].strip()

        # Markdown emphasis-only action lines are not dialogue.
        if cleaned.startswith("*") or cleaned.endswith("*"):
            return ""

        # Heuristic: narration/action endings with no question mark are likely not speech.
        if re.search(r"(하며|바라보며|고개를|시선을|확인한다|지켜본다|생각한다)[.。]?$", cleaned):
            return ""

        return ""

    @staticmethod
    def _clean_action_text(text: str) -> str:
        """Make compact action text from raw line."""
        out = str(text or "")
        out = re.sub(r"\[\[.*?\]\]|\[.*?\]", "", out)
        out = out.replace("*", "")
        out = re.sub(r"\s+", " ", out).strip()
        return out

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

            start_turn = self._coerce_turn_number(chunk[0].get("turn", 0), default=0)
            end_turn = self._coerce_turn_number(chunk[-1].get("turn", 0), default=start_turn)
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
            content = self._compact_turn_content(ix)
            # Truncate very long content
            if len(content) > 250:
                content = content[:247] + "..."
            action_type = ix.get("action_type", "dialogue")
            tag = "[SCENE]" if action_type == "director_event" else ""
            lines.append(f"T{turn} {tag}{speaker}: {content}")
        return "\n".join(lines)

    def _compact_turn_content(self, ix: dict) -> str:
        """Reduce simulation formatting noise before distillation."""
        content = str(ix.get("content", "") or "")
        action_type = str(ix.get("action_type", "dialogue") or "dialogue")

        if action_type == "dialogue":
            spoken = self._extract_spoken_dialogue(content)
            if spoken:
                return f"\"{spoken}\""
            # Fallback to cleaned text if quote extraction fails.
            return self._clean_action_text(content)

        if action_type in ("action", "inner_thought"):
            return self._clean_action_text(content)

        return content

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        """Parse a JSON array from LLM response."""
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for key in ("scenes", "items", "data"):
                    value = result.get(key)
                    if isinstance(value, list):
                        return value
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

    @staticmethod
    def _coerce_turn_number(value, default: int = 0) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, (list, tuple)):
            first = value[0] if value else default
            return SceneDistiller._coerce_turn_number(first, default=default)
        raw = str(value or "")
        if not raw.strip():
            return default
        raw = raw.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")).strip()
        match = re.search(r"(?<!\d)(\d{1,5})(?!\d)", raw)
        if not match:
            return default
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_string_list(value) -> list[str]:
        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]
        if isinstance(value, str):
            parts = re.split(r"[,/|]\s*|\n+", value)
            return [part.strip() for part in parts if part.strip()]
        return []

    @staticmethod
    def _coerce_dialogue_rows(value) -> list[dict]:
        if isinstance(value, dict):
            value = [value]
        elif isinstance(value, str):
            value = [value]
        elif not isinstance(value, list):
            return []

        rows: list[dict] = []
        for raw in value:
            if isinstance(raw, dict):
                speaker = str(raw.get("speaker", "") or raw.get("name", "")).strip() or "Unknown"
                line = str(raw.get("line", "") or raw.get("content", "")).strip()
            else:
                text = str(raw or "").strip()
                if not text:
                    continue
                match = re.match(r"^\s*([^:：]{1,40})\s*[:：]\s*(.+?)\s*$", text)
                if match:
                    speaker = match.group(1).strip() or "Unknown"
                    line = match.group(2).strip()
                else:
                    speaker = "Unknown"
                    line = text
            if not line:
                continue
            rows.append({"speaker": speaker, "line": line})
        return rows

    def _normalize_scene_payload(self, raw_scene) -> dict:
        if not isinstance(raw_scene, dict):
            return {}
        return {
            "title": str(raw_scene.get("title", "")).strip(),
            "turn_start": raw_scene.get("turn_start", 0),
            "turn_end": raw_scene.get("turn_end", 0),
            "turn_range": raw_scene.get("turn_range"),
            "location": str(raw_scene.get("location", "")).strip(),
            "characters": self._coerce_string_list(
                raw_scene.get("characters", raw_scene.get("characters_present", []))
            ),
            "key_dialogue": self._coerce_dialogue_rows(raw_scene.get("key_dialogue", [])),
            "key_actions": self._coerce_string_list(raw_scene.get("key_actions", [])),
            "discoveries": self._coerce_string_list(raw_scene.get("discoveries", [])),
            "emotional_arc": str(raw_scene.get("emotional_arc", "")).strip(),
            "beat_refs": self._coerce_string_list(
                raw_scene.get("beat_refs", raw_scene.get("beat_references", []))
            ),
            "summary": str(
                raw_scene.get("summary", raw_scene.get("narrative_summary", ""))
            ).strip(),
            "pacing": str(raw_scene.get("pacing", "building")).strip() or "building",
        }

    def _coerce_scene_turn_range(
        self,
        payload: dict,
        available_turns: list[int],
        scene_index: int,
        total_scenes: int,
    ) -> tuple[int, int]:
        fallback_start, fallback_end = self._fallback_scene_turn_range(
            available_turns,
            scene_index,
            total_scenes,
        )
        raw_range = payload.get("turn_range")
        if isinstance(raw_range, str) and raw_range.strip():
            parts = re.findall(r"\d{1,5}", raw_range.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")))
            raw_start = parts[0] if parts else payload.get("turn_start", fallback_start)
            raw_end = parts[1] if len(parts) >= 2 else parts[0] if parts else payload.get("turn_end", fallback_end)
        elif isinstance(raw_range, (list, tuple)) and raw_range:
            raw_start = raw_range[0]
            raw_end = raw_range[1] if len(raw_range) >= 2 else raw_range[0]
        else:
            raw_start = payload.get("turn_start", fallback_start)
            raw_end = payload.get("turn_end", fallback_end)

        start = self._coerce_turn_number(raw_start, default=fallback_start)
        end = self._coerce_turn_number(raw_end, default=fallback_end)
        if available_turns:
            start = max(available_turns[0], min(available_turns[-1], start))
            end = max(available_turns[0], min(available_turns[-1], end))
        if start > end:
            start, end = end, start
        return start, end

    def _fallback_scene_turn_range(
        self,
        available_turns: list[int],
        scene_index: int,
        total_scenes: int,
    ) -> tuple[int, int]:
        if not available_turns:
            base = max(1, scene_index + 1)
            return base, base
        total = max(1, total_scenes)
        start_idx = min(
            len(available_turns) - 1,
            (scene_index * len(available_turns)) // total,
        )
        end_idx = min(
            len(available_turns) - 1,
            max(start_idx, ((scene_index + 1) * len(available_turns) - 1) // total),
        )
        return available_turns[start_idx], available_turns[end_idx]

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for v in values:
            if not isinstance(v, str):
                continue
            vv = v.strip()
            if not vv or vv in seen:
                continue
            seen.add(vv)
            out.append(vv)
        return out

    def _dedupe_semantic_lines(self, values: list[str], limit: int) -> list[str]:
        """
        Remove near-duplicate lines that differ only by small surface variation.
        Keeps the most concise representation first and preserves order.
        """
        ordered = self._dedupe_preserve_order(list(values or []))
        kept: list[str] = []
        fingerprints: list[set[str]] = []

        for line in ordered:
            fp = self._line_fingerprint_set(line)
            if not fp:
                continue
            if any(self._fingerprint_overlap(fp, prev) >= 0.78 for prev in fingerprints):
                continue
            kept.append(line)
            fingerprints.append(fp)
            if len(kept) >= max(1, limit):
                break
        return kept

    @staticmethod
    def _line_fingerprint_set(text: str) -> set[str]:
        cleaned = str(text or "").lower()
        cleaned = re.sub(r"[^0-9a-z가-힣\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        stop = {"그리고", "하지만", "그러나", "그는", "그녀는", "the", "and", "for"}
        tokens = [t for t in cleaned.split() if len(t) >= 2 and t not in stop]
        return set(tokens[:18])

    @staticmethod
    def _fingerprint_overlap(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)

    @staticmethod
    def _token_overlap_score(a: str, b: str) -> float:
        toks_a = set(re.findall(r"[A-Za-z가-힣0-9\\-]{2,}", (a or "").lower()))
        toks_b = set(re.findall(r"[A-Za-z가-힣0-9\\-]{2,}", (b or "").lower()))
        if not toks_a or not toks_b:
            return 0.0
        return len(toks_a & toks_b) / len(toks_a | toks_b)
