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
            if isinstance(para_sent_cap, int):
                lines.append(f"- Keep scene summary paragraphs under about {para_sent_cap} sentence(s).")
            if isinstance(jargon_cap, int):
                lines.append(f"- Keep technical/jargon concepts to about {jargon_cap} per paragraph.")
            if isinstance(dense_cap, int):
                lines.append(f"- Compress dense explanatory runs into <= {dense_cap} sentence chunks.")
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
        for i, sd in enumerate(scenes_data):
            raw_start = int(sd.get("turn_start", 0) or 0)
            raw_end = int(sd.get("turn_end", 0) or 0)
            turn_start, turn_end = (raw_start, raw_end) if raw_start <= raw_end else (raw_end, raw_start)
            scene = DistilledScene(
                scene_number=i + 1,
                title=sd.get("title", f"Scene {i + 1}"),
                turn_range=(turn_start, turn_end),
                location=sd.get("location", ep_location),
                characters_present=sd.get("characters", []),
                key_dialogue=sd.get("key_dialogue", []),
                key_actions=sd.get("key_actions", []),
                discoveries=sd.get("discoveries", []),
                emotional_arc=sd.get("emotional_arc", ""),
                beat_references=sd.get("beat_refs", []),
                narrative_summary=sd.get("summary", ""),
                pacing=sd.get("pacing", "building"),
                raw_turn_count=max(1, turn_end - turn_start + 1),
            )
            self._sanitize_scene_dialogue(scene)
            self._compress_expository_dialogue(scene)
            scene.narrative_summary = self._tighten_narrative_summary(scene.narrative_summary)
            scene.narrative_summary = self._rebalance_narrative_summary(scene)
            scene.characters_present = self._canonicalize_name_list(
                scene.characters_present,
                canonical_speakers,
            )
            for row in scene.key_dialogue:
                if not isinstance(row, dict):
                    continue
                row["speaker"] = self._canonicalize_name(
                    str(row.get("speaker", "")).strip(),
                    canonical_speakers,
                )
            scene.key_actions = self._dedupe_semantic_lines(scene.key_actions, limit=8)
            scene.discoveries = self._dedupe_semantic_lines(scene.discoveries, limit=6)
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
            turn = int(ix.get("turn", 0) or 0)
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

        return scenes

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

    def _reader_prefers_faster_progression(self) -> bool:
        corpus = self._reader_feedback_corpus()
        return any(
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
        return any(
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

    def _scene_merge_budget(self, target_scenes: int) -> int:
        budget = 0
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
            if best_idx < 0 or best_score < 4:
                break
            merged[best_idx: best_idx + 2] = [
                self._merge_scene_pair(merged[best_idx], merged[best_idx + 1])
            ]
            merges_left -= 1

        return merged

    def _adjacent_scene_merge_score(self, left: DistilledScene, right: DistilledScene) -> int:
        score = 0
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
        if self._token_overlap_score(left.narrative_summary, right.narrative_summary) >= 0.35:
            score += 2
        if self._summary_is_mood_heavy(left.narrative_summary) or self._summary_is_mood_heavy(right.narrative_summary):
            score += 1
        if left.pacing == right.pacing:
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
        return " ".join(parts[:max_sentences]).strip()

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
        return " ".join(sentences[:max_sentences]).strip()

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
        return " ".join(keep[:max_sentences]).strip()

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
        return self._ensure_summary_sentence(f"{base}, 그 여파로 {follow}")

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
