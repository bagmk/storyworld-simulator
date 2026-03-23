"""
Chapter polishing stage for the novel-generation pipeline.

This stage is intentionally bounded:
  - one broad polish pass
  - one anchor-coverage correction pass when needed
  - one reader-feedback final pass when needed
  - deterministic cleanup/normalization only

It should not invent beats, reorder scenes, or reinterpret scene structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Pre-compiled patterns used in structural repair and redundancy scan
_RE_PARAGRAPH_SPLIT = re.compile(r'\n{2,}')
_RE_KOREAN_SENTENCE_SPLIT = re.compile(
    r'(?<=[다요죠]\.)\s+|(?<=[다요죠]\!)\s+|(?<=[다요죠]\?)\s+'
)

from .llm_client import LLMClient
from .reader_profile import ReaderProfile, build_reader_profile
from .review_feedback import build_feedback_prompt_block
from .scene_state import (
    detect_abstract_tension_without_consequence,
    count_stock_body_cues,
    count_stock_connective_openers,
    count_abstract_tension_nouns,
    STOCK_CONNECTIVES,
)

# Korean suspension-ending patterns used in _pre_polish_redundancy_scan
_SUSPENSION_ENDINGS = (
    "었다.", "있었다.", "들었다.", "느꼈다.", "생각했다.",
    "스쳤다.", "맴돌았다.", "울렸다.",
)


@dataclass(frozen=True)
class ChapterPolishProfile:
    repeat_terms: tuple[str, ...]
    jargon_terms: tuple[str, ...]
    term_repeat_cap: int
    sentence_word_cap: int
    paragraph_sentence_cap: Optional[int]
    dense_sentence_cap: int
    jargon_term_cap: int
    sensory_channel_cap: int
    prefers_sentence_simplification: bool
    prefers_technical_term_restraint: bool
    prefers_single_term_gloss: bool
    prefers_stable_term_reuse: bool
    prefers_list_breakup: bool
    prefers_explicit_transition_cues: bool
    prefers_dialogue_compaction: bool
    prefers_observable_emotion_evidence: bool
    prefers_expository_dialogue_reduction: bool
    prefers_analytical_wording_reduction: bool
    wants_emotional_wave_contrast: bool
    wants_distinct_dialogue_voices: bool
    needs_role_cues: bool
    needs_contextual_summaries: bool
    needs_draft_cleanup: bool
    prefers_sumin_first_person: bool
    reports_stalled_progression: bool
    needs_reader_feedback_pass: bool


class ChapterPolisher:
    def __init__(
        self,
        llm: LLMClient,
        episode_config: dict,
        runtime_policy: Optional[dict] = None,
        reader_feedback: Optional[dict] = None,
    ) -> None:
        self.llm = llm
        self.episode_config = episode_config or {}
        self.runtime_policy = runtime_policy or {}
        self.reader_profile: ReaderProfile = build_reader_profile(reader_feedback)
        self.reader_feedback = self.reader_profile.as_dict()
        self.chapter_profile = self._build_chapter_profile()

    def _build_chapter_profile(self) -> ChapterPolishProfile:
        profile = self.reader_profile
        prefers_sentence_simplification = profile.prefers_sentence_simplification()
        prefers_technical_term_restraint = profile.prefers_technical_term_restraint()
        prefers_single_term_gloss = profile.prefers_single_term_gloss()
        prefers_stable_term_reuse = profile.prefers_stable_term_reuse()
        prefers_list_breakup = profile.prefers_list_breakup()
        prefers_explicit_transition_cues = profile.prefers_explicit_transition_cues()
        prefers_dialogue_compaction = profile.prefers_dialogue_compaction()
        prefers_observable_emotion_evidence = profile.prefers_observable_emotion_evidence()
        prefers_expository_dialogue_reduction = profile.prefers_expository_dialogue_reduction()
        prefers_analytical_wording_reduction = profile.prefers_analytical_wording_reduction()
        wants_emotional_wave_contrast = profile.wants_emotional_wave_contrast()
        wants_distinct_dialogue_voices = profile.wants_distinct_dialogue_voices()
        needs_role_cues = profile.needs_role_cues()
        needs_contextual_summaries = profile.needs_contextual_summaries()
        needs_draft_cleanup = profile.needs_draft_cleanup()
        prefers_sumin_first_person = profile.prefers_sumin_first_person()
        reports_stalled_progression = profile.reports_stalled_progression()

        needs_reader_feedback_pass = any(
            (
                prefers_sentence_simplification,
                prefers_technical_term_restraint,
                prefers_stable_term_reuse,
                prefers_list_breakup,
                prefers_explicit_transition_cues,
                prefers_dialogue_compaction,
                prefers_observable_emotion_evidence,
                prefers_expository_dialogue_reduction,
                prefers_analytical_wording_reduction,
                wants_emotional_wave_contrast,
                wants_distinct_dialogue_voices,
                needs_role_cues,
                needs_contextual_summaries,
                needs_draft_cleanup,
                prefers_sumin_first_person,
                reports_stalled_progression,
            )
        )

        return ChapterPolishProfile(
            repeat_terms=tuple(profile.repeat_terms(max_terms=6)),
            jargon_terms=tuple(profile.jargon_terms(max_terms=6)),
            term_repeat_cap=profile.term_repeat_cap(default=2),
            sentence_word_cap=profile.sentence_word_cap(default=25),
            paragraph_sentence_cap=profile.paragraph_sentence_cap(),
            dense_sentence_cap=profile.dense_sentence_cap(default=2),
            jargon_term_cap=profile.jargon_term_cap(default=2),
            sensory_channel_cap=profile.sensory_channel_cap(default=2),
            prefers_sentence_simplification=prefers_sentence_simplification,
            prefers_technical_term_restraint=prefers_technical_term_restraint,
            prefers_single_term_gloss=prefers_single_term_gloss,
            prefers_stable_term_reuse=prefers_stable_term_reuse,
            prefers_list_breakup=prefers_list_breakup,
            prefers_explicit_transition_cues=prefers_explicit_transition_cues,
            prefers_dialogue_compaction=prefers_dialogue_compaction,
            prefers_observable_emotion_evidence=prefers_observable_emotion_evidence,
            prefers_expository_dialogue_reduction=prefers_expository_dialogue_reduction,
            prefers_analytical_wording_reduction=prefers_analytical_wording_reduction,
            wants_emotional_wave_contrast=wants_emotional_wave_contrast,
            wants_distinct_dialogue_voices=wants_distinct_dialogue_voices,
            needs_role_cues=needs_role_cues,
            needs_contextual_summaries=needs_contextual_summaries,
            needs_draft_cleanup=needs_draft_cleanup,
            prefers_sumin_first_person=prefers_sumin_first_person,
            reports_stalled_progression=reports_stalled_progression,
            needs_reader_feedback_pass=needs_reader_feedback_pass,
        )

    def polish_chapter(
        self,
        text: str,
        target_words: int,
        style: str,
        protagonist_name: str,
        chapter_anchors: Optional[list[str]],
        prose_adapter,
    ) -> str:
        # ── Structural scan before LLM polish (Part 8) ───────────────────────
        if self.runtime_policy.get("structural_repair_before_polish", True):
            text = self._structural_repair_pass(text, prose_adapter)

        polished = self.run_llm_polish(text, target_words, style, chapter_anchors, prose_adapter)
        polished = self.ensure_anchor_coverage(
            polished,
            chapter_anchors or [],
            target_words,
            style,
            prose_adapter,
        )
        polished = self.apply_reader_feedback_pass(
            polished,
            target_words,
            style,
            chapter_anchors,
            prose_adapter,
        )
        return self.apply_deterministic_cleanup(
            polished,
            style,
            protagonist_name,
            prose_adapter,
        )

    def apply_deterministic_cleanup(
        self,
        text: str,
        style: str,
        protagonist_name: str,
        prose_adapter,
    ) -> str:
        if not text:
            return text

        profile = self.chapter_profile
        cleaned = prose_adapter._enforce_pov_timeline_guards(text, style, protagonist_name)
        cleaned = prose_adapter._cleanup_pov_reference_artifacts(cleaned, style, protagonist_name)
        cleaned = prose_adapter._enforce_jargon_onboarding_and_variation(cleaned)
        cleaned = prose_adapter._reduce_local_repetition(cleaned)
        cleaned = prose_adapter._compress_repeated_tension_beats(cleaned)
        cleaned = prose_adapter._trim_post_metaphor_explanations(cleaned)
        cleaned = prose_adapter._trim_redundant_sensory_sentences(cleaned)
        cleaned = prose_adapter._trim_redundant_emotion_sentences(cleaned)
        cleaned = prose_adapter._diversify_transition_openers(cleaned)
        cleaned = prose_adapter._merge_clipped_sentence_runs(cleaned)
        cleaned = prose_adapter._stagger_sentence_rhythm(cleaned)
        cleaned = prose_adapter._diversify_transition_openers(cleaned)
        cleaned = prose_adapter._compress_redundant_jargon_sentences(cleaned)
        cleaned = prose_adapter._enforce_sentence_word_caps(
            cleaned,
            max_words=profile.sentence_word_cap,
        )
        cleaned = prose_adapter._insert_short_beats_after_long_streak(
            cleaned,
            long_threshold=22,
            streak_limit=2,
        )
        cleaned = prose_adapter._strengthen_dialogue_action_beats(cleaned)
        cleaned = prose_adapter._split_dense_information_paragraphs(cleaned)
        cleaned = prose_adapter._cap_paragraph_term_repetition(
            cleaned,
            max_per_paragraph=profile.term_repeat_cap,
        )
        cleaned = prose_adapter._apply_sensory_diversity_guard(cleaned, recent_window=3)
        if prose_adapter._effective_style(style) == "third_person_close":
            cleaned = prose_adapter._reinforce_name_refresh(cleaned, protagonist_name)
        prose_adapter._warn_sensory_streak(cleaned, streak_limit=3)
        prose_adapter._log_paragraph_split_recommendations(cleaned)
        return prose_adapter._normalize_paragraphs(cleaned)

    def _pre_polish_redundancy_scan(self, text: str) -> str:
        """
        Python-side structural scan before LLM polish pass.
        Returns a guidance string to include in the polish prompt.
        Checks: suspension run, abstract tension without consequence,
        stock connective overuse, stock body cue excess.
        """
        import logging
        _log = logging.getLogger(__name__)

        paragraphs = [p.strip() for p in _RE_PARAGRAPH_SPLIT.split(text) if p.strip()]
        if len(paragraphs) < 3:
            return ""

        guidance_parts: list[str] = []

        # 1. Adjacent paragraphs ending on same suspension pattern
        suspension_run = 0
        max_suspension_run = 0
        for p in paragraphs:
            if any(p.endswith(e) for e in _SUSPENSION_ENDINGS):
                suspension_run += 1
                max_suspension_run = max(max_suspension_run, suspension_run)
            else:
                suspension_run = 0
        if max_suspension_run >= 3:
            guidance_parts.append(
                f"WARNING: {max_suspension_run} consecutive paragraphs end on the same suspension pattern. "
                "End at least one on action, consequence, decision, or physical movement."
            )

        # 2. Abstract tension without nearby concrete consequence
        flagged_tension = detect_abstract_tension_without_consequence(paragraphs, window=2)
        if flagged_tension:
            _log.info(
                "[Polisher] Abstract tension without consequence in %d para(s): %s",
                len(flagged_tension), flagged_tension[:5],
            )
            guidance_parts.append(
                f"WARNING: {len(flagged_tension)} paragraph(s) contain repeated abstract danger/pressure/choice language "
                "without a concrete consequence nearby. After each abstract tension beat, show one concrete result: "
                "blocked access, document reveal, person exit, changed room state, or forced decision."
            )

        # 3. Stock connective opener overuse
        conn_counts = count_stock_connective_openers(paragraphs)
        heavy_conns = {k: v for k, v in conn_counts.items() if v >= 3}
        if heavy_conns:
            _log.info("[Polisher] Stock connective opener overuse: %s", heavy_conns)
            conn_str = ", ".join(f"'{k}'(×{v})" for k, v in heavy_conns.items())
            guidance_parts.append(
                f"WARNING: Stock connective openers overused: {conn_str}. "
                "Replace at least half with gaze shifts, footsteps, object cues, or paragraph-opening subject names."
            )

        # 4. Stock body cue excess (cap per chapter = 3×scene cap)
        body_counts = count_stock_body_cues(text)
        body_cap = int(self.runtime_policy.get("repeated_body_cue_cap_per_scene", 2)) * 3
        heavy_body = {k: v for k, v in body_counts.items() if v > body_cap}
        if heavy_body:
            _log.info("[Polisher] Stock body cues exceeding chapter cap (%d): %s", body_cap, heavy_body)
            body_str = ", ".join(f"'{k}'(×{v})" for k, v in list(heavy_body.items())[:5])
            guidance_parts.append(
                f"WARNING: Repeated stock gesture cues: {body_str}. "
                "Vary or remove later occurrences — substitute with room reaction, object movement, or posture shift."
            )

        # 5. Abstract tension noun overuse
        noun_counts = count_abstract_tension_nouns(text)
        noun_cap = int(self.runtime_policy.get("abstract_tension_noun_cap_per_scene", 4)) * 2
        over_noun = {k: v for k, v in noun_counts.items() if v > noun_cap}
        if over_noun:
            _log.info("[Polisher] Abstract tension nouns over chapter cap: %s", over_noun)
            noun_str = ", ".join(f"'{k}'(×{v})" for k, v in list(over_noun.items())[:5])
            guidance_parts.append(
                f"WARNING: Abstract tension nouns overused: {noun_str}. "
                "Replace later occurrences with a concrete institutional or physical anchor."
            )

        return "\n".join(guidance_parts)

    def _structural_repair_pass(self, text: str, prose_adapter) -> str:
        """
        Deterministic structural repair before LLM polish.

        Repairs that can be done reliably without LLM:
        1. Deduplicate near-identical adjacent paragraph openings (subject reset)
        2. Cap stock connective openers (replace excess with blank paragraph break)
        3. Remove exact duplicate sentences within a 5-sentence window
        """
        import logging
        _log = logging.getLogger(__name__)

        if not text:
            return text

        paragraphs = [p.strip() for p in _RE_PARAGRAPH_SPLIT.split(text) if p.strip()]
        if len(paragraphs) < 2:
            return text

        # 1. Remove exact duplicate adjacent sentences within paragraphs
        repaired: list[str] = []
        for para in paragraphs:
            sentences = _RE_KOREAN_SENTENCE_SPLIT.split(para)
            seen: list[str] = []
            for sent in sentences:
                sent_stripped = sent.strip()
                if not sent_stripped:
                    continue
                # Exact duplicate of immediately previous sentence → skip
                if seen and sent_stripped == seen[-1]:
                    _log.info("[Polisher] Removed exact duplicate sentence in paragraph.")
                    continue
                seen.append(sent_stripped)
            repaired.append(" ".join(seen))

        # 2. Cap stock connective openers — if 3+ paragraphs start with same connector,
        #    replace the 3rd+ occurrences by removing the opener (keep rest of sentence)
        for conn in STOCK_CONNECTIVES:
            count = 0
            for i, para in enumerate(repaired):
                if para.startswith(conn):
                    count += 1
                    if count >= 3:
                        # Remove opening connective
                        stripped = para[len(conn):].lstrip(" ,.")
                        if stripped:
                            repaired[i] = stripped[0].upper() + stripped[1:] if stripped else stripped
                            _log.info(
                                "[Polisher] Removed excess stock connective opener '%s' from paragraph %d.",
                                conn, i,
                            )

        return "\n\n".join(repaired)

    def run_llm_polish(
        self,
        text: str,
        target_words: int,
        style: str,
        chapter_anchors: Optional[list[str]],
        prose_adapter,
    ) -> str:
        current = len(text.split())
        profile = self.chapter_profile
        style = prose_adapter._effective_style(style)
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
            f"- Paragraphs should usually contain {prose_adapter._readability_controls()['paragraph_min']}-{prose_adapter._readability_controls()['paragraph_max']} sentences\n"
            f"- Most sentences should stay under about {profile.sentence_word_cap} words; split explanatory chains early\n"
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
            f"- If a concept recurs (coherence/drift/latency classes), keep one stable core term and vary the surrounding sentence shape or consequence instead of synonym-swapping the term itself\n"
            f"- If 3+ consecutive sentences use same sensory channel, switch channel (sound/touch/temperature)\n"
            f"- If explanatory rhythm grows monotonous, use one grounded action/reaction sentence instead of a detached fragment\n"
            f"- If an anchor term is missing, introduce it once verbatim when context allows; later callbacks may be shorter as long as the evidence stays clear: {anchors_text}\n"
            f"- If any anchor is missing, add it naturally without changing core events\n\n"
        )
        if profile.prefers_stable_term_reuse:
            prompt += "- Keep one stable term per concept; avoid synonym swapping for the same idea\n"
        if profile.needs_contextual_summaries or profile.prefers_list_breakup:
            prompt += "- End dense information paragraphs with one short takeaway summary sentence\n"
        if profile.sensory_channel_cap <= 2:
            prompt += "- Keep sensory detail focused to 1-2 sensory channels per paragraph\n"
        if profile.needs_role_cues or profile.wants_distinct_dialogue_voices:
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
        redundancy_guidance = self._pre_polish_redundancy_scan(text)
        if redundancy_guidance:
            prompt += f"## Pre-Polish Redundancy Notes\n{redundancy_guidance}\n\n"
        prompt += f"Full chapter text:\n\n{text}"

        polished = self.llm.chat(
            [{"role": "user", "content": prompt}],
            purpose="prose_polish",
            use_premium=True,
            temperature=float(self.runtime_policy.get("prose_polish_temperature", 0.4) or 0.4),
            max_tokens=min(16000, max(6000, target_words * 5)),
        )
        return prose_adapter._normalize_paragraphs(polished)

    def ensure_anchor_coverage(
        self,
        text: str,
        chapter_anchors: list[str],
        target_words: int,
        style: str,
        prose_adapter,
    ) -> str:
        if not text or not chapter_anchors:
            return text

        profile = self.chapter_profile
        anchors = [a.strip() for a in chapter_anchors if isinstance(a, str) and len(a.strip()) >= 3][:30]
        if not anchors:
            return text

        present = [anchor for anchor in anchors if anchor.lower() in text.lower()]
        required_present = min(5, max(2, len(anchors) // 5))
        if profile.prefers_technical_term_restraint:
            required_present = min(required_present, max(1, len(anchors) // 6))
        if target_words < 2200:
            required_present = min(required_present, 2)
        if len(present) >= required_present:
            return text

        missing_cap = 3 if profile.prefers_technical_term_restraint else 6
        missing = [anchor for anchor in anchors if anchor not in present][:missing_cap]
        style = prose_adapter._effective_style(style)
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
        return prose_adapter._normalize_paragraphs(revised)

    def apply_reader_feedback_pass(
        self,
        text: str,
        target_words: int,
        style: str,
        chapter_anchors: Optional[list[str]],
        prose_adapter,
    ) -> str:
        if not text or not self.reader_feedback:
            return text

        profile = self.chapter_profile
        if not profile.needs_reader_feedback_pass:
            return text

        style = prose_adapter._effective_style(style)
        pov = "first person" if style == "first_person" else "third person close"
        anchors = chapter_anchors or []
        anchors_text = ", ".join(anchors[:20]) if anchors else "(none)"
        review_guidance = build_feedback_prompt_block(self.reader_feedback, max_items=6)
        repeat_terms = list(profile.repeat_terms)
        jargon_terms = list(profile.jargon_terms)
        repeat_term_line = (
            f"- 독자 반복 지적 단어({', '.join(repeat_terms[:6])})는 문단당 과다 반복 금지\n"
            if repeat_terms else ""
        )
        jargon_term_line = (
            f"- 독자 난해 지적 기술어({', '.join(jargon_terms[:6])})는 첫 언급에만 짧게 풀고 재등장은 축약\n"
            if jargon_terms else ""
        )
        jargon_density_cap = profile.jargon_term_cap
        dense_sentence_cap = profile.dense_sentence_cap
        paragraph_sentence_cap = profile.paragraph_sentence_cap
        sentence_cap = profile.sentence_word_cap
        paragraph_cap_line = (
            f"- 문단은 최대 {paragraph_sentence_cap}문장까지 유지하고 초과 시 분할\n"
            if paragraph_sentence_cap is not None else ""
        )
        first_person_cleanup_line = (
            "- 수민 서술은 1인칭으로 고정하고 내레이션에서 '수민은/그는' 식 자기지칭을 제거할 것\n"
            if style == "first_person" else ""
        )
        draft_cleanup_line = (
            "- 미완 문장, 대명사 오류, 호칭·지시어 혼선을 먼저 정리해 초안 흔적을 지울 것\n"
            if profile.needs_draft_cleanup else ""
        )
        extra_constraints: list[str] = []
        if profile.needs_role_cues or profile.wants_distinct_dialogue_voices:
            extra_constraints.append("- 대화 구간은 1-2회 발화마다 누가 말하는지 드러나게 정리")
        if profile.prefers_expository_dialogue_reduction:
            extra_constraints.append("- 정보 전달형 대사는 짧게 압축하고, 바로 행동/표정/침묵 반응을 붙여 임팩트를 살릴 것")
        if profile.prefers_dialogue_compaction:
            extra_constraints.append("- 긴 회의/대화 구간은 연속 설명 대사를 줄이고 행동/환경 반응 비트를 교차 배치할 것")
        if profile.prefers_observable_emotion_evidence:
            extra_constraints.append("- 설명적 심리문이 길면 행동/표정/반응 단서로 치환해 감정을 보여줄 것")
            extra_constraints.append("- 영어 키워드/기술 용어 뒤에는 바로 인물의 이해, 당혹, 긴장, 행동 반응을 붙일 것")
        if profile.sensory_channel_cap <= 2:
            extra_constraints.append("- 비슷한 감각 묘사와 심리 표현은 같은 문단/인접 문단에서 반복하지 말 것")
        if profile.wants_emotional_wave_contrast:
            extra_constraints.append("- 감정 강도는 단조롭게 유지하지 말고 짧은 완화 비트 후 다시 긴장을 세울 것")
        if profile.prefers_explicit_transition_cues:
            extra_constraints.append("- 장소/장면 전환 지점은 한 줄 전환 문장으로 연결해 흐름을 명확히 할 것")
        if profile.prefers_analytical_wording_reduction:
            extra_constraints.append("- 가능성/계산/추론 같은 분석 어휘는 반복하지 말고 한 번만 압축적으로 사용")
        if profile.prefers_stable_term_reuse:
            extra_constraints.append("- 같은 개념에는 하나의 안정된 용어를 유지하고 동의어 교체를 줄일 것")
        if profile.prefers_technical_term_restraint or profile.prefers_single_term_gloss:
            extra_constraints.append("- 기술 용어/약어 뒤의 풀이는 첫 1회만 짧게 두고 이후 설명 재반복은 피할 것")
        extra_block = "".join(f"{line}\n" for line in extra_constraints)

        prompt = (
            "다음 한국어 소설 본문을 사건/정보/감정선 순서를 유지한 채 1회 리라이트하라.\n"
            "핵심 목적: 독자 리뷰 반영(반복 축소, 문단 호흡 개선, 기술 용어 과밀 완화, 화자 명확성 강화).\n\n"
            "제약:\n"
            f"- 시점 유지: {pov}\n"
            f"- 분량: 약 {target_words}단어 근처 유지\n"
            "- 동일 정보/표현의 반복은 삭제 또는 통합\n"
            "- 같은 기능의 문단이 이어지면 하나로 압축하고 사건 축을 더 곧게 세울 것\n"
            "- 질문→응답→접근/제안 순으로 장면을 정렬하고, 이미 지나간 단계로 되감지 말 것\n"
            "- 긴 문단은 1-2문장 단위로 자연 분할\n"
            f"{paragraph_cap_line}"
            f"- 대부분의 문장은 약 {sentence_cap}어절 이하로 유지하고, 길어지면 인과 단위로 분리\n"
            "- 같은 문장 구조나 문장 시작 패턴이 이어지면 하나 이상 변형해 리듬을 바꿀 것\n"
            "- '그리고', '그러자', '다만', '그 직후', '잠시 뒤' 같은 연결어를 연속 문장 시작에 반복하지 말 것\n"
            "- 장면 전환은 시선 이동, 걸음, 문, 마이크, 의자 같은 물리적 신호로 처리하고 시간 부사 남용은 줄일 것\n"
            "- 짧은 문장이 연속될 때는 누가/왜/어디서가 보이도록 연결해 문맥을 보강할 것\n"
            "- 같은 장면의 2~3개 단문이 한 박자로 이어지면 하나의 복합문으로 자연스럽게 묶을 것\n"
            "- 강한 문장과 담백한 문장을 섞어 압박의 고저를 만들 것\n"
            "- 같은 정보나 해석이 이미 한 번 전달됐다면 다음 문장에서는 되풀이하지 말고 반응, 행동, 결정으로 넘어갈 것\n"
            "- 기술 용어/약어는 꼭 필요할 때만 짧게 풀고 이후는 짧은 콜백으로 유지할 것\n"
            "- latency, real-time 같은 기술어는 첫 1회만 짧게 풀고 이후에는 수민의 판단, 감정, 선택을 전면에 둘 것\n"
            "- 괄호 설명은 기본값으로 쓰지 말고, 필요하면 본문 안에 짧게 녹여 쓸 것\n"
            "- 어려운 기술 개념은 필요할 때만 짧은 일상 비교를 한 번 붙이고 바로 장면 행동으로 돌아갈 것\n"
            f"- 문단당 기술 용어는 최대 {jargon_density_cap}개 내에서 유지(초과 개념은 통합/요약)\n"
            f"- 정보량이 많은 설명 문장은 최대 {dense_sentence_cap}문장으로 압축\n"
            f"{repeat_term_line}"
            f"{jargon_term_line}"
            "- 한 문장에는 동작, 감정, 판단 중 한 축만 남기고 필요하면 원인과 결과를 나눌 것\n"
            "- 비유를 쓴 직후 그 의미를 다시 설명하는 문장은 삭제하거나 행동/반응으로 치환할 것\n"
            "- 각 문단은 상황 변화, 압박 상승, 발견 중 하나를 분명히 남겨 전개를 전진시킬 것\n"
            "- 이미 알려진 인물을 매번 새 호칭으로 재소개하지 말 것\n"
            "- 인물의 역할/의도는 장면상 필요할 때만 짧게 제시하고 중복 설명은 삭제\n"
            f"{first_person_cleanup_line}"
            f"{draft_cleanup_line}"
            f"{extra_block}"
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

    # Compatibility shims for older call sites/tests.
    def _llm_polish(
        self,
        text: str,
        target_words: int,
        style: str,
        chapter_anchors: Optional[list[str]],
        prose_adapter,
    ) -> str:
        return self.run_llm_polish(text, target_words, style, chapter_anchors, prose_adapter)

    def _ensure_anchor_coverage(
        self,
        text: str,
        chapter_anchors: list[str],
        target_words: int,
        style: str,
        prose_adapter,
    ) -> str:
        return self.ensure_anchor_coverage(
            text,
            chapter_anchors,
            target_words,
            style,
            prose_adapter,
        )

    def _reader_feedback_final_pass(
        self,
        text: str,
        target_words: int,
        style: str,
        chapter_anchors: Optional[list[str]],
        prose_adapter,
    ) -> str:
        return self.apply_reader_feedback_pass(
            text,
            target_words,
            style,
            chapter_anchors,
            prose_adapter,
        )
