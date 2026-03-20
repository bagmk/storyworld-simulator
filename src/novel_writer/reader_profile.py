"""
Centralized reader-feedback interpretation for the novel-generation pipeline.

This module is intentionally profile-driven: callers should prefer querying a
single ReaderProfile over re-implementing keyword scans and style-cap parsing
in each stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Optional


MAX_WORDS_PER_SCENE = 1000

PATTERN_REGISTRY: dict[str, tuple[str, ...]] = {
    "stalled_progression": (
        "멈춘 이유",
        "멈춘",
        "멈춤",
        "정체",
        "제자리",
        "맴도는",
        "전진감이 약",
        "안 나가",
        "진행이 안",
        "흐름이 끊",
    ),
    "draft_cleanup": (
        "영어 혼입",
        "영어 표현",
        "오탈자",
        "퇴고 전 원고",
        "원고처럼 보이게",
        "real-time",
        "real-time viable if externally supported",
        "수민는",
        "단어은",
        "미완 문장",
        "대명사 오류",
        "호칭 혼선",
        "지시어 혼선",
    ),
    "first_person_preference": (
        "수민 1인칭",
        "수민 1인칭 시점",
        "1인칭 시점",
        "1인칭으로",
        "1인칭 고정",
        "시점으로 고정",
    ),
    "compact_beats": (
        "기술",
        "용어",
        "약어",
        "약자",
        "전문",
        "jargon",
        "acronym",
        "반복",
        "중복",
        "리스트",
        "목록",
        "나열",
        "정보 전달",
    ),
    "faster_progression": (
        "전개가 느려",
        "느려서 집중",
        "집중력을 잃",
        "늘어지",
        "같은 장면을 다시 보는",
        "다른 문장으로 다시 보는",
        "제자리에서 맴도",
        "서사적 전진감",
        "템포가 느려",
        "속도감이 떨어",
    ),
    "repeated_confrontation_merge": (
        "복도 대면",
        "밀러와의 복도 대면",
        "밀러 접촉",
        "하나의 대화로 압축",
        "질문의 강도",
        "제자리에서 다시 시작",
        "사실상 두 번 반복",
    ),
    "timeline_confusion": (
        "시간축",
        "시간 순서",
        "순서가 섞",
        "되감기",
        "헷갈리",
        "복도 대치",
        "발표 종료 후",
        "질의응답",
        "슬라이드 발표",
        "단선 구조",
    ),
    "contextual_summaries": (
        "간결한 문장",
        "문맥 파악",
        "맥락 파악",
        "따라가기 힘들",
        "문맥이 어려",
    ),
    "stronger_scene_compaction": (
        "반복되는 표현",
        "비슷한 상황",
        "비슷한 상황과 묘사",
        "묘사가 반복",
        "같은 정보와 감정",
        "재진술",
        "다른 문장으로 다시 보는",
        "제자리에서 맴도",
        "문장이 너무 길",
        "길고 복잡",
        "이해하기 어려",
        "지루",
    ),
    "recycled_negotiation_points": (
        "밀러와의 대화",
        "협상 논점",
        "핵심 조건",
        "외부 지원",
        "실시간성",
        "통제권",
        "책임 문제",
        "여러 차례 되풀이",
    ),
    "stock_bridge_phrases": (
        "짧은 숨이 스친 뒤",
        "반복 접속구",
        "호흡 문구",
        "문장 리듬이 기계적",
        "그 말이 끝나자",
        "시선이 옮겨가자",
        "비슷한 리듬",
    ),
}


@dataclass(frozen=True)
class ReaderSemanticFlags:
    stalled_progression: bool
    needs_draft_cleanup: bool
    prefers_sumin_first_person: bool
    prefers_compact_beats: bool
    prefers_faster_progression: bool
    wants_repeated_confrontation_merge: bool
    reports_timeline_confusion: bool
    needs_contextual_summaries: bool
    prefers_stronger_scene_compaction: bool
    flags_recycled_negotiation_points: bool
    flags_stock_bridge_phrases: bool
    force_reaction_after_jargon: bool
    summary_plain_buffer_enabled: bool
    summary_easy_metaphor_enabled: bool


@dataclass(frozen=True)
class ReaderCaps:
    term_repeat_cap: int
    sentence_word_cap: int
    paragraph_sentence_cap: Optional[int]
    dense_sentence_cap: int
    jargon_term_cap: int
    sensory_channel_cap: int
    emotion_repeat_cap: int
    transition_char_window: tuple[int, int]
    short_beat_char_window: tuple[int, int]
    short_beats_per_scene: tuple[int, int]
    transition_opener_cap: int
    sentence_variety_window: int
    summary_sentence_word_cap: int
    static_threat_signal_cap: int
    scene_compaction_target: int
    tension_phrase_cap: int


@dataclass(frozen=True)
class ReaderStageHints:
    distiller_prefers_compaction: bool
    distiller_prioritize_chronology: bool
    prose_force_first_person: bool
    prose_needs_draft_cleanup: bool
    chapter_reduce_scene_count: bool


@dataclass(frozen=True)
class ReaderTermPreferences:
    repetition_watch_terms: tuple[str, ...]
    jargon_watch_terms: tuple[str, ...]
    transition_avoid_terms: tuple[str, ...]


def _feedback_corpus(review: dict) -> str:
    if not isinstance(review, dict) or not review:
        return ""
    parts: list[str] = []
    for key in ("what_felt_boring_or_hard", "style_tips"):
        vals = review.get(key, []) or []
        if isinstance(vals, list):
            parts.extend(str(v) for v in vals if str(v).strip())
    parts.append(str(review.get("reader_comment", "") or ""))
    return " ".join(parts).lower()


def _feedback_has_any(review: dict, *tokens: str) -> bool:
    corpus = _feedback_corpus(review)
    return bool(corpus) and any(str(token).lower() in corpus for token in tokens if token)


def _reader_feedback_mentions_stalled_progression(reader_feedback: dict) -> bool:
    return _feedback_has_any(reader_feedback, *PATTERN_REGISTRY["stalled_progression"])


def _reader_feedback_needs_draft_cleanup(reader_feedback: dict) -> bool:
    return _feedback_has_any(reader_feedback, *PATTERN_REGISTRY["draft_cleanup"])


def _reader_feedback_prefers_sumin_first_person(reader_feedback: dict) -> bool:
    return _feedback_has_any(reader_feedback, *PATTERN_REGISTRY["first_person_preference"])


def _merge_transition_avoid_terms(constraints: dict, *terms: str) -> None:
    existing = constraints.get("avoid_transition_terms", [])
    if isinstance(existing, str):
        existing_list = [existing]
    elif isinstance(existing, list):
        existing_list = [str(term) for term in existing if str(term).strip()]
    else:
        existing_list = []
    seen = {term.strip().lower() for term in existing_list if term.strip()}
    for term in terms:
        cleaned = str(term or "").strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        existing_list.append(cleaned)
        seen.add(cleaned.lower())
    constraints["avoid_transition_terms"] = existing_list


def apply_reader_feedback_pipeline_overrides(reader_feedback: dict) -> dict:
    if not isinstance(reader_feedback, dict) or not reader_feedback:
        return reader_feedback if isinstance(reader_feedback, dict) else {}

    tuned = dict(reader_feedback)
    constraints = dict(tuned.get("style_constraints", {}) or {})
    changed = False

    if _feedback_has_any(
        tuned,
        "짧게 끊기는 문장",
        "문장이 너무 자주 끊기",
        "짧은 반복 문장",
        "비슷한 리듬",
        "같은 리듬",
        "단조로운 리듬",
        "그 말이 끝나자",
        "시선이 옮겨가자",
    ):
        constraints["short_beats_per_scene_min"] = 0
        try:
            existing_max = int(constraints.get("short_beats_per_scene_max", 1))
        except (TypeError, ValueError):
            existing_max = 1
        constraints["short_beats_per_scene_max"] = max(0, min(1, existing_max))
        try:
            short_min = int(constraints.get("short_beat_chars_min", 0))
        except (TypeError, ValueError):
            short_min = 0
        try:
            short_max = int(constraints.get("short_beat_chars_max", 0))
        except (TypeError, ValueError):
            short_max = 0
        constraints["short_beat_chars_min"] = max(14, short_min)
        constraints["short_beat_chars_max"] = max(
            28, short_max, constraints["short_beat_chars_min"]
        )
        constraints["sentence_variety_window"] = max(
            4,
            int(constraints.get("sentence_variety_window", 4) or 4),
        )
        changed = True

    if _feedback_has_any(
        tuned,
        "기술 용어",
        "기술 용어가 자주",
        "기술 용어가 겹칠 때",
        "기술 용어가 연달아",
        "영어 표현",
        "영어 키워드",
        "추상 표현",
        "고등학생",
        "한 번에 이해",
        "괄호 설명",
        "설명문",
        "브리핑 문서",
        "약어",
        "약자",
    ):
        try:
            jargon_cap = int(constraints.get("max_jargon_terms_per_paragraph", 2))
        except (TypeError, ValueError):
            jargon_cap = 2
        constraints["max_jargon_terms_per_paragraph"] = (
            1
            if _feedback_has_any(tuned, "고등학생", "한 번에 이해", "추상 표현")
            else max(1, min(2, jargon_cap))
        )
        try:
            dense_cap = int(constraints.get("max_sentences_in_dense_info", 2))
        except (TypeError, ValueError):
            dense_cap = 2
        constraints["max_sentences_in_dense_info"] = max(1, min(2, dense_cap))
        constraints["jargon_buffer_sentences"] = max(
            1,
            int(constraints.get("jargon_buffer_sentences", 1) or 1),
        )
        constraints["force_reaction_after_jargon"] = 1
        constraints["summary_easy_metaphor_once"] = 0
        try:
            summary_words_cap = int(
                constraints.get("scene_summary_sentence_words_max", 15) or 15
            )
        except (TypeError, ValueError):
            summary_words_cap = 15
        constraints["scene_summary_sentence_words_max"] = min(15, summary_words_cap)
        changed = True

    if _feedback_has_any(
        tuned,
        "설명문",
        "장면을 따라가기보다 설명문",
        "용어 설명",
        "상황 해석",
        "상황 해석이 잦",
        "설명과 해석",
        "반응과 행동으로 보여",
        "이미 한 번 이해된 개념",
        "반복 해설하지 말고",
        "감각 변화",
        "선택 압박",
    ):
        constraints["max_jargon_terms_per_paragraph"] = 1
        constraints["max_sentences_in_dense_info"] = 1
        constraints["force_reaction_after_jargon"] = 1
        constraints["summary_easy_metaphor_once"] = 0
        changed = True

    if _feedback_has_any(
        tuned,
        "비슷한 감각 묘사",
        "감각 묘사",
        "심리 표현",
        "비슷한 정보",
        "비슷한 감정",
        "같은 장면을 맴도",
        "같은 장면을 다시 도는",
        "긴장 묘사",
        "긴장감",
        "압박",
        "반복되는 표현",
        "묘사가 반복",
    ):
        constraints["max_term_repeats_per_scene"] = 1
        constraints["tension_phrase_cap"] = 1
        constraints["max_sensory_channels_per_paragraph"] = 2
        constraints["max_emotion_repeats_per_scene"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "같은 긴장을 여러 번",
        "같은 긴장",
        "비슷한 뜻의 문장",
        "비슷한 문장으로 여러 번",
        "같은 정보를 여러 번",
    ):
        constraints["max_term_repeats_per_scene"] = 1
        constraints["tension_phrase_cap"] = 1
        constraints["max_emotion_repeats_per_scene"] = 1
        changed = True

    if _reader_feedback_mentions_stalled_progression(tuned):
        try:
            paragraph_cap = int(constraints.get("max_sentences_per_paragraph", 2) or 2)
        except (TypeError, ValueError):
            paragraph_cap = 2
        constraints["max_sentences_per_paragraph"] = min(2, paragraph_cap)
        try:
            dense_cap = int(constraints.get("max_sentences_in_dense_info", 2) or 2)
        except (TypeError, ValueError):
            dense_cap = 2
        constraints["max_sentences_in_dense_info"] = min(2, dense_cap)
        try:
            summary_words_cap = int(
                constraints.get("scene_summary_sentence_words_max", 15) or 15
            )
        except (TypeError, ValueError):
            summary_words_cap = 15
        constraints["scene_summary_sentence_words_max"] = min(14, summary_words_cap)
        constraints["tension_phrase_cap"] = 1
        constraints["max_emotion_repeats_per_scene"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "쉼표",
        "연결어",
        "쉼표와 접속",
        "문장이 너무 길",
        "길고 복잡",
        "읽는 속도",
        "가독성",
        "호흡",
        "걸리는",
    ):
        try:
            sentence_chars_max = int(constraints.get("sentence_chars_max", 60))
        except (TypeError, ValueError):
            sentence_chars_max = 60
        constraints["sentence_chars_max"] = max(48, min(56, sentence_chars_max))
        constraints["scene_summary_sentence_words_max"] = 14
        try:
            paragraph_cap = int(constraints.get("max_sentences_per_paragraph", 2) or 2)
        except (TypeError, ValueError):
            paragraph_cap = 2
        constraints["max_sentences_per_paragraph"] = min(2, paragraph_cap)
        changed = True

    if _feedback_has_any(
        tuned,
        "짧은 문장과 긴 문장을 섞",
        "속도감과 긴장감",
        "대화 장면의 속도감",
        "문장 길이를 다양",
    ):
        constraints["sentence_variety_window"] = max(
            5,
            int(constraints.get("sentence_variety_window", 5) or 5),
        )
        try:
            short_max = int(constraints.get("short_beats_per_scene_max", 1) or 1)
        except (TypeError, ValueError):
            short_max = 1
        constraints["short_beats_per_scene_max"] = max(1, min(2, short_max))
        changed = True

    if _feedback_has_any(
        tuned,
        "손가락",
        "숨",
        "노트북",
        "반응 묘사",
        "인위적으로",
    ):
        constraints["reaction_motif_repeat_cap"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "그리고",
        "그러자",
        "이어서",
        "그 순간",
        "접속 습관",
        "문장 연결",
        "연결이 너무 자주",
        "연결어 사용 빈도",
        "더 자연스럽",
        "덜 작위적",
        "그 말이 끝나자",
        "시선이 옮겨가자",
        "연결어 반복",
    ):
        constraints["max_transition_openers_per_block"] = 1
        _merge_transition_avoid_terms(constraints, "그리고", "그러자", "이어서", "그 순간")
        changed = True

    if _feedback_has_any(
        tuned,
        "한 문장에는 동작이나 감정 한 축",
        "한 문장에는",
        "문장 연결 방식을 수정",
        "그리고, 그러자",
        "그리고 그러자",
        "리듬을 날카롭게",
        "호흡이 무거",
    ):
        constraints["single_axis_sentences"] = 1
        constraints["max_transition_openers_per_block"] = 1
        _merge_transition_avoid_terms(constraints, "그리고", "그러자", "이어서", "그 순간")
        changed = True

    if _feedback_has_any(
        tuned,
        "그 말이 끝나자",
        "시선이 옮겨가자",
        "연결어 반복",
        "상투적 연결어",
    ):
        constraints["max_transition_openers_per_block"] = 1
        _merge_transition_avoid_terms(
            constraints,
            "그리고",
            "그러자",
            "이어서",
            "그 순간",
            "그 말이 끝나자",
            "시선이 옮겨가자",
            "고개를 들자",
            "의자가 밀리자",
        )
        changed = True

    if _feedback_has_any(
        tuned,
        "짧은 숨이 스친 뒤",
        "반복 접속구",
        "호흡 문구",
        "문장 리듬이 기계적",
    ):
        constraints["max_transition_openers_per_block"] = 1
        constraints["sentence_variety_window"] = max(
            5,
            int(constraints.get("sentence_variety_window", 5) or 5),
        )
        _merge_transition_avoid_terms(constraints, "짧은 숨이 스친 뒤")
        changed = True

    if _feedback_has_any(
        tuned,
        "비유로 분위기를 만든 직후 의미를 다시 설명",
        "의미를 다시 설명",
        "비유",
        "문단 밀도",
        "호흡이 가벼워",
    ):
        constraints["avoid_metaphor_explanation"] = 1
        constraints["summary_easy_metaphor_once"] = 0
        changed = True

    if _feedback_has_any(
        tuned,
        "메타 표식",
        "작업 메모",
        "ep01의 온도계",
        "ep01—scene21",
        "완성 원고가 아니라 작업 메모",
    ):
        constraints["strip_meta_markers"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "외부 지원",
        "실시간성",
        "자원과 통제",
        "통제권",
        "책임 문제",
        "후반 반복",
        "20퍼센트",
        "20%",
        "압축",
    ):
        constraints["scene_compaction_ratio_target"] = 80
        constraints["max_term_repeats_per_scene"] = 1
        constraints["max_emotion_repeats_per_scene"] = 1
        constraints["dialogue_agenda_contrast"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "밀러와의 대화",
        "협상 논점",
        "핵심 조건",
        "여러 차례 되풀이",
    ):
        try:
            compaction_target = int(constraints.get("scene_compaction_ratio_target", 80) or 80)
        except (TypeError, ValueError):
            compaction_target = 80
        constraints["scene_compaction_ratio_target"] = min(80, compaction_target)
        constraints["dialogue_agenda_contrast"] = 1
        constraints["merge_repeated_confrontation_beats"] = 1
        constraints["prefer_linear_scene_axis"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "같은 정보와 감정이 재진술",
        "재진술",
        "제자리에서 맴도",
        "다른 문장으로 다시 보는",
        "서사적 전진감",
        "후반 반복",
    ):
        constraints["scene_compaction_ratio_target"] = 75
        constraints["max_term_repeats_per_scene"] = 1
        constraints["max_emotion_repeats_per_scene"] = 1
        constraints["single_strong_interior_beat"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "복도 대면",
        "밀러와의 복도 대면",
        "밀러 접촉",
        "사실상 두 번 반복",
        "하나의 대화로 압축",
        "질문의 강도",
        "제자리에서 다시 시작",
    ):
        try:
            compaction_target = int(constraints.get("scene_compaction_ratio_target", 75) or 75)
        except (TypeError, ValueError):
            compaction_target = 75
        constraints["scene_compaction_ratio_target"] = min(75, compaction_target)
        constraints["merge_repeated_confrontation_beats"] = 1
        constraints["prefer_linear_scene_axis"] = 1
        constraints["clarify_event_transitions"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "메모 발견",
        "경고음",
        "밀러 등장",
        "장면 전환",
        "공간 동선",
        "인물 위치",
    ):
        constraints["clarify_event_transitions"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "위협 신호",
        "모니터 경보",
        "보안요원 시선",
        "과밀",
        "인위적으로",
    ):
        constraints["clarify_event_transitions"] = 1
        constraints["compress_threat_signal_stack"] = 1
        constraints["max_static_threat_signals_per_scene"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "시간축",
        "시간 순서",
        "순서가 섞",
        "되감기",
        "헷갈리",
        "복도 대치",
        "발표 종료 후",
        "질의응답",
        "슬라이드 발표",
        "단선 구조",
        "질문→응답→접근",
        "질문->응답->접근",
    ):
        constraints["clarify_event_transitions"] = 1
        constraints["prioritize_chronological_scene_order"] = 1
        constraints["prefer_linear_scene_axis"] = 1
        constraints["scene_compaction_ratio_target"] = 75
        constraints["max_transition_openers_per_block"] = 1
        _merge_transition_avoid_terms(
            constraints,
            "그리고",
            "그러자",
            "이어서",
            "그 순간",
            "그 직후",
            "잠시 뒤",
        )
        changed = True

    if _reader_feedback_prefers_sumin_first_person(tuned):
        try:
            refresh_streak = int(constraints.get("speaker_refresh_streak", 2) or 2)
        except (TypeError, ValueError):
            refresh_streak = 2
        constraints["force_first_person_pov"] = 1
        constraints["speaker_refresh_streak"] = max(2, refresh_streak)
        changed = True

    if _feedback_has_any(
        tuned,
        "미완 문장",
        "대명사 오류",
        "호칭 혼선",
        "지시어 혼선",
        "퇴고 전 초안",
        "퇴고 전 원고",
        "신뢰도를 떨어",
    ):
        constraints["force_complete_sentences"] = 1
        constraints["stabilize_reference_labels"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "다크 수트 남자",
        "크리스찬 밀러",
        "같은 인물인지",
        "다른 인물인지",
        "헷갈린다",
    ):
        constraints["clarify_similar_character_entries"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "길게 호흡",
        "핵심 문단",
        "문단 몇 개는 길게",
    ):
        constraints["prefer_pivot_paragraph_breath"] = 1
        constraints["sentence_variety_window"] = max(
            5,
            int(constraints.get("sentence_variety_window", 5) or 5),
        )
        changed = True

    if _reader_feedback_needs_draft_cleanup(tuned):
        constraints["max_jargon_terms_per_paragraph"] = 1
        constraints["force_reaction_after_jargon"] = 1
        constraints["force_complete_sentences"] = 1
        constraints["stabilize_reference_labels"] = 1
        changed = True

    if _feedback_has_any(
        tuned,
        "모레노",
        "밀러",
        "이해관계",
        "말버릇",
        "대사는 테마 설명보다",
    ):
        constraints["dialogue_agenda_contrast"] = 1
        changed = True

    if changed:
        tuned["style_constraints"] = constraints
    return tuned


def sanitize_chapter_draft_artifacts(chapter_text: str, reader_feedback: dict) -> str:
    if not chapter_text or not _reader_feedback_needs_draft_cleanup(reader_feedback):
        return chapter_text

    cleaned = str(chapter_text)
    replacements = {
        "real-time viable if externally supported": "외부 지원이 붙을 때만 실시간 대응이 가능했다",
        "수민는": "수민은",
        "단어은": "단어는",
    }
    for bad, good in replacements.items():
        cleaned = re.sub(re.escape(bad), good, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\breal[- ]time\b", "실시간", cleaned, flags=re.IGNORECASE)
    return cleaned


def resolve_generation_style(cli_style: str, reader_feedback: dict) -> str:
    style = str(cli_style or "third_person_close").strip() or "third_person_close"
    if style != "first_person" and _reader_feedback_prefers_sumin_first_person(reader_feedback):
        return "first_person"
    return style


def adjust_scene_target_for_feedback(
    target_scenes: int,
    target_words: int,
    reader_feedback: dict,
    *,
    max_words_per_scene: int = MAX_WORDS_PER_SCENE,
) -> int:
    corpus = _feedback_corpus(reader_feedback)
    if not corpus:
        return target_scenes

    adjusted = target_scenes
    if any(token in corpus for token in ("전개가 느려", "느려서 집중", "집중력을 잃", "늘어지", "템포가 느려", "속도감이 떨어")):
        adjusted -= 1
    if target_words <= 4500 and adjusted >= 5 and _reader_feedback_mentions_stalled_progression(reader_feedback):
        adjusted -= 1
    if (
        target_words <= 4000
        and adjusted >= 5
        and any(token in corpus for token in ("간결한 문장", "문맥 파악", "맥락 파악", "따라가기 힘들", "문맥이 약"))
    ):
        adjusted -= 1
    if (
        target_words <= 4500
        and adjusted >= 6
        and _feedback_has_any(
            reader_feedback,
            "반복되는 표현",
            "비슷한 상황",
            "비슷한 상황과 묘사",
            "같은 정보와 감정이 재진술",
            "재진술",
            "같은 장면을 다시 도는",
            "같은 장면을 맴도",
            "제자리에서 맴도",
            "복도 대면",
            "밀러와의 복도 대면",
            "사실상 두 번 반복",
            "하나의 대화로 압축",
            "다른 문장으로 다시 보는",
            "묘사가 반복",
            "문장이 너무 길",
            "길고 복잡",
            "이해하기 어려",
        )
    ):
        adjusted -= 1
    if (
        target_words <= 4500
        and adjusted >= 5
        and _feedback_has_any(
            reader_feedback,
            "짧게 끊기는 문장",
            "문장이 너무 자주 끊기",
            "비슷한 리듬",
            "같은 리듬",
            "단조로운 리듬",
            "그 말이 끝나자",
            "시선이 옮겨가자",
            "연결어 반복",
            "기술 용어가 자주",
            "기술 용어가 겹칠 때",
        )
    ):
        adjusted -= 1
    if (
        target_words <= 4500
        and adjusted >= 5
        and _feedback_has_any(
            reader_feedback,
            "같은 긴장을 여러 번",
            "같은 긴장",
            "비슷한 뜻의 문장",
            "비슷한 문장으로 여러 번",
            "설명문을 읽는",
            "장면을 따라가기보다 설명문",
            "용어 설명",
            "상황 해석이 잦",
        )
    ):
        adjusted -= 1
    if (
        target_words <= 4500
        and adjusted >= 4
        and _feedback_has_any(
            reader_feedback,
            "시간축",
            "시간 순서",
            "순서가 섞",
            "되감기",
            "헷갈리",
            "복도 대치",
            "발표 종료 후",
            "질의응답",
            "슬라이드 발표",
            "단선 구조",
        )
    ):
        adjusted -= 1
    if (
        target_words <= 4500
        and adjusted >= 5
        and _feedback_has_any(
            reader_feedback,
            "외부 지원",
            "실시간성",
            "자원과 통제",
            "통제권",
            "책임 문제",
            "후반 반복",
            "20퍼센트",
            "20%",
            "압축",
        )
    ):
        adjusted -= 1
    if (
        target_words <= 4500
        and adjusted >= 5
        and _feedback_has_any(
            reader_feedback,
            "위협 신호",
            "모니터 경보",
            "보안요원 시선",
            "과밀",
            "인위적으로",
        )
    ):
        adjusted -= 1

    scene_floor = math.ceil(target_words / max_words_per_scene)
    return max(scene_floor, adjusted)


@dataclass
class ReaderProfile:
    feedback: dict
    raw_feedback: dict = field(init=False, repr=False)
    corpus: str = field(init=False)
    semantic_flags: ReaderSemanticFlags = field(init=False)
    caps: ReaderCaps = field(init=False)
    stage_hints: ReaderStageHints = field(init=False)
    term_preferences: ReaderTermPreferences = field(init=False)
    pattern_registry: dict[str, tuple[str, ...]] = field(init=False, repr=False)
    normalization_metadata: dict[str, object] = field(init=False, repr=False)

    @classmethod
    def from_feedback(cls, reader_feedback: Optional[dict]) -> "ReaderProfile":
        return cls(dict(reader_feedback) if isinstance(reader_feedback, dict) else {})

    def __post_init__(self) -> None:
        self.raw_feedback = dict(self.feedback) if isinstance(self.feedback, dict) else {}
        normalized = apply_reader_feedback_pipeline_overrides(dict(self.raw_feedback))
        self.feedback = dict(normalized) if isinstance(normalized, dict) else {}
        self.corpus = _feedback_corpus(self.feedback)
        self.pattern_registry = dict(PATTERN_REGISTRY)
        self.term_preferences = self._build_term_preferences()
        self.caps = self._build_caps()
        self.semantic_flags = self._build_semantic_flags()
        self.stage_hints = self._build_stage_hints()
        self.normalization_metadata = {
            "applied_pipeline_overrides": self.feedback != self.raw_feedback,
            "style_constraint_keys": tuple(sorted(self.style_constraints().keys())),
            "has_feedback": bool(self.corpus),
        }

    def as_dict(self) -> dict:
        return dict(self.feedback)

    def raw_as_dict(self) -> dict:
        return dict(self.raw_feedback)

    def _build_term_preferences(self) -> ReaderTermPreferences:
        return ReaderTermPreferences(
            repetition_watch_terms=tuple(self.repeat_terms(max_terms=10)),
            jargon_watch_terms=tuple(self.jargon_terms(max_terms=10)),
            transition_avoid_terms=tuple(sorted(self.transition_avoid_terms())),
        )

    def _build_caps(self) -> ReaderCaps:
        return ReaderCaps(
            term_repeat_cap=self.term_repeat_cap(default=2),
            sentence_word_cap=self.sentence_word_cap(default=25),
            paragraph_sentence_cap=self.paragraph_sentence_cap(),
            dense_sentence_cap=self.dense_sentence_cap(default=2),
            jargon_term_cap=self.jargon_term_cap(default=2),
            sensory_channel_cap=self.sensory_channel_cap(default=2),
            emotion_repeat_cap=self.emotion_repeat_cap(default=1),
            transition_char_window=self.transition_char_window(),
            short_beat_char_window=self.short_beat_char_window(),
            short_beats_per_scene=self.short_beats_per_scene(),
            transition_opener_cap=self.transition_opener_cap(default=2),
            sentence_variety_window=self.sentence_variety_window(default=4),
            summary_sentence_word_cap=self.summary_sentence_word_cap(default=18),
            static_threat_signal_cap=self.static_threat_signal_cap(default=2),
            scene_compaction_target=self.scene_compaction_target(default=100),
            tension_phrase_cap=self.tension_phrase_cap(default=2),
        )

    def _build_semantic_flags(self) -> ReaderSemanticFlags:
        return ReaderSemanticFlags(
            stalled_progression=self.reports_stalled_progression(),
            needs_draft_cleanup=self.needs_draft_cleanup(),
            prefers_sumin_first_person=self.prefers_sumin_first_person(),
            prefers_compact_beats=self.prefers_compact_beats(),
            prefers_faster_progression=self.prefers_faster_progression(),
            wants_repeated_confrontation_merge=self.wants_repeated_confrontation_merge(),
            reports_timeline_confusion=self.reports_timeline_confusion(),
            needs_contextual_summaries=self.needs_contextual_summaries(),
            prefers_stronger_scene_compaction=self.prefers_stronger_scene_compaction(),
            flags_recycled_negotiation_points=self.flags_recycled_negotiation_points(),
            flags_stock_bridge_phrases=self.flags_stock_bridge_phrases(),
            force_reaction_after_jargon=self.force_reaction_after_jargon(),
            summary_plain_buffer_enabled=self.summary_plain_buffer_enabled(),
            summary_easy_metaphor_enabled=self.summary_easy_metaphor_enabled(),
        )

    def _build_stage_hints(self) -> ReaderStageHints:
        return ReaderStageHints(
            distiller_prefers_compaction=(
                self.prefers_compact_beats()
                or self.prefers_stronger_scene_compaction()
                or self.wants_repeated_confrontation_merge()
            ),
            distiller_prioritize_chronology=(
                self.reports_timeline_confusion()
                or self.flag_enabled("prioritize_chronological_scene_order")
            ),
            prose_force_first_person=self.prefers_sumin_first_person(),
            prose_needs_draft_cleanup=self.needs_draft_cleanup(),
            chapter_reduce_scene_count=(
                self.reports_stalled_progression()
                or self.prefers_faster_progression()
                or self.prefers_stronger_scene_compaction()
            ),
        )

    def has_any(self, *tokens: str) -> bool:
        return _feedback_has_any(self.feedback, *tokens)

    def mentions(self, *keywords: str) -> bool:
        if not self.feedback:
            return False
        all_text = self.corpus
        lowered = [str(k).lower() for k in keywords if k]
        if any(k in all_text for k in lowered):
            return True

        if any(k in lowered for k in ("긴 문장", "문장이 길", "긴 문단", "문단이 길", "문단", "호흡", "리듬", "속도감", "정보가 밀집", "밀집", "길게 느껴", "길고 복잡", "이해하기 어려", "이해하기 어렵")):
            if any(token in all_text for token in ("비슷한 리듬", "같은 리듬", "단조", "단조롭", "단조롭게", "리듬이 반복", "속도감이 단조", "속도감이 떨어", "템포가 느려", "템포가 떨어", "길고 복잡", "이해하기 어려", "이해하기 어렵")):
                return True

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

        if any(k in lowered for k in ("누구의 말", "누가 말", "누가 누구", "화자", "대사 구분", "헷갈", "인물", "역할", "구분", "호칭", "이름", "말투", "어투", "톤", "speaker")):
            if any(token in all_text for token in ("초반", "따라가기 힘들", "맥락", "인물 설명 없이", "누군지")):
                return True

        if any(k in lowered for k in ("정보 전달형 대사", "정보 전달", "설명 위주", "감정적 임팩트", "임팩트")):
            if any(token in all_text for token in ("정보 전달 위주", "설명 위주", "감정의 고저", "감정 고저", "대화가 대부분", "건조", "긴 회의", "회의·대화", "대화 장면", "대사가 계속")):
                return True

        if any(k in lowered for k in ("감정선", "감정의 고저", "감정 고저", "감정의 파고", "긴장 완화", "유머", "친근한 묘사")):
            if any(token in all_text for token in ("감정의 파고", "감정 파고", "감정의 고저", "감정 고저", "긴장 완화", "작은 유머", "유머", "친근한 묘사")):
                return True
        return False

    def style_constraints(self) -> dict:
        raw = self.feedback.get("style_constraints", {})
        return raw if isinstance(raw, dict) else {}

    def flag_enabled(self, key: str, default: bool = False) -> bool:
        constraints = self.style_constraints()
        raw = constraints.get(key, 1 if default else 0)
        try:
            enabled = int(raw)
        except (TypeError, ValueError):
            return default
        return enabled >= 1

    def repeat_terms(self, max_terms: int = 10) -> list[str]:
        terms = self.feedback.get("repetition_watch_terms", []) or []
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
        return out[:max_terms]

    def jargon_terms(self, max_terms: int = 10) -> list[str]:
        terms = self.feedback.get("jargon_watch_terms", []) or []
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
            if len(out) >= max_terms:
                break
        return out

    def term_repeat_cap(self, default: int = 2) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("max_term_repeats_per_scene")
        if raw is None:
            raw = constraints.get("max_term_repeats_per_paragraph", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(5, cap))

    def sentence_word_cap(self, default: int = 25) -> int:
        constraints = self.style_constraints()
        raw_hi = constraints.get("sentence_chars_max")
        try:
            hi = int(raw_hi)
        except (TypeError, ValueError):
            return default
        return max(10, min(default, int(round(hi / 3.2))))

    def paragraph_sentence_cap(self) -> Optional[int]:
        constraints = self.style_constraints()
        raw = constraints.get("max_sentences_per_paragraph")
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            return None
        return max(1, min(8, cap))

    def dense_sentence_cap(self, default: int = 2) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("max_sentences_in_dense_info", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(4, cap))

    def jargon_term_cap(self, default: int = 2) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("max_jargon_terms_per_paragraph", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(8, cap))

    def sensory_channel_cap(self, default: int = 2) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("max_sensory_channels_per_paragraph", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(3, cap))

    def emotion_repeat_cap(self, default: int = 1) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("max_emotion_repeats_per_scene", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(3, cap))

    def transition_char_window(self) -> tuple[int, int]:
        constraints = self.style_constraints()
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

    def short_beat_char_window(self) -> tuple[int, int]:
        constraints = self.style_constraints()
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

    def short_beats_per_scene(self) -> tuple[int, int]:
        constraints = self.style_constraints()
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

    def transition_opener_cap(self, default: int = 2) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("max_transition_openers_per_block", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(4, cap))

    def transition_avoid_terms(self) -> set[str]:
        constraints = self.style_constraints()
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

    def sentence_variety_window(self, default: int = 4) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("sentence_variety_window", default)
        try:
            window = int(raw)
        except (TypeError, ValueError):
            window = default
        return max(3, min(6, window))

    def summary_sentence_word_cap(self, default: int = 18) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("scene_summary_sentence_words_max", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(10, min(20, cap))

    def static_threat_signal_cap(self, default: int = 2) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("max_static_threat_signals_per_scene", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(3, cap))

    def scene_compaction_target(self, default: int = 100) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("scene_compaction_ratio_target", default)
        try:
            ratio = int(raw)
        except (TypeError, ValueError):
            ratio = default
        return max(60, min(100, ratio))

    def tension_phrase_cap(self, default: int = 2) -> int:
        constraints = self.style_constraints()
        raw = constraints.get("tension_phrase_cap", default)
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            cap = default
        return max(1, min(4, cap))

    def force_reaction_after_jargon(self) -> bool:
        return self.flag_enabled("force_reaction_after_jargon")

    def summary_plain_buffer_enabled(self) -> bool:
        constraints = self.style_constraints()
        raw = constraints.get("jargon_buffer_sentences", 1)
        try:
            enabled = int(raw)
        except (TypeError, ValueError):
            enabled = 1
        return enabled >= 1

    def summary_easy_metaphor_enabled(self) -> bool:
        constraints = self.style_constraints()
        raw = constraints.get("summary_easy_metaphor_once", 1)
        try:
            enabled = int(raw)
        except (TypeError, ValueError):
            enabled = 1
        return enabled >= 1

    def needs_draft_cleanup(self) -> bool:
        return self.mentions(
            "영어 혼입",
            "영어 표현",
            "오탈자",
            "퇴고 전 초안",
            "퇴고 전 원고",
            "미완 문장",
            "대명사 오류",
            "호칭 혼선",
            "지시어 혼선",
        ) or self.flag_enabled("force_complete_sentences") or self.flag_enabled("stabilize_reference_labels")

    def prefers_sumin_first_person(self) -> bool:
        return _reader_feedback_prefers_sumin_first_person(self.feedback)

    def resolve_generation_style(self, cli_style: str) -> str:
        return resolve_generation_style(cli_style, self.feedback)

    def sanitize_chapter_draft_artifacts(self, chapter_text: str) -> str:
        return sanitize_chapter_draft_artifacts(chapter_text, self.feedback)

    def adjusted_scene_target(
        self,
        target_scenes: int,
        target_words: int,
        *,
        max_words_per_scene: int = MAX_WORDS_PER_SCENE,
    ) -> int:
        return adjust_scene_target_for_feedback(
            target_scenes,
            target_words,
            self.feedback,
            max_words_per_scene=max_words_per_scene,
        )

    def reports_stalled_progression(self) -> bool:
        return _reader_feedback_mentions_stalled_progression(self.feedback)

    def prefers_compact_beats(self) -> bool:
        return self.has_any(
            "기술", "용어", "약어", "약자", "전문", "jargon", "acronym",
            "반복", "중복", "리스트", "목록", "나열", "정보 전달",
        )

    def prefers_faster_progression(self) -> bool:
        return self.reports_stalled_progression() or self.wants_repeated_confrontation_merge() or self.has_any(
            "전개가 느려",
            "느려서 집중",
            "집중력을 잃",
            "늘어지",
            "같은 장면을 다시 보는",
            "다른 문장으로 다시 보는",
            "제자리에서 맴도",
            "서사적 전진감",
            "템포가 느려",
            "속도감이 떨어",
        )

    def wants_repeated_confrontation_merge(self) -> bool:
        return self.flag_enabled("merge_repeated_confrontation_beats") or self.has_any(
            "복도 대면",
            "밀러와의 복도 대면",
            "밀러 접촉",
            "하나의 대화로 압축",
            "질문의 강도",
            "제자리에서 다시 시작",
            "사실상 두 번 반복",
        )

    def reports_timeline_confusion(self) -> bool:
        return self.has_any(
            "시간축",
            "시간 순서",
            "순서가 섞",
            "되감기",
            "헷갈리",
            "복도 대치",
            "발표 종료 후",
            "질의응답",
            "슬라이드 발표",
            "단선 구조",
        )

    def needs_contextual_summaries(self) -> bool:
        return self.has_any(
            "간결한 문장",
            "문맥 파악",
            "맥락 파악",
            "따라가기 힘들",
            "문맥이 어려",
        )

    def prefers_stronger_scene_compaction(self) -> bool:
        return self.reports_stalled_progression() or self.wants_repeated_confrontation_merge() or self.scene_compaction_target() <= 85 or self.has_any(
            "반복되는 표현",
            "비슷한 상황",
            "비슷한 상황과 묘사",
            "묘사가 반복",
            "같은 정보와 감정",
            "재진술",
            "다른 문장으로 다시 보는",
            "제자리에서 맴도",
            "문장이 너무 길",
            "길고 복잡",
            "이해하기 어려",
            "지루",
        )

    def flags_recycled_negotiation_points(self) -> bool:
        return self.has_any(
            "밀러와의 대화",
            "협상 논점",
            "핵심 조건",
            "외부 지원",
            "실시간성",
            "통제권",
            "책임 문제",
            "여러 차례 되풀이",
        )

    def flags_stock_bridge_phrases(self) -> bool:
        return self.has_any(
            "짧은 숨이 스친 뒤",
            "반복 접속구",
            "호흡 문구",
            "문장 리듬이 기계적",
            "그 말이 끝나자",
            "시선이 옮겨가자",
            "비슷한 리듬",
        )

    def prefers_explicit_transition_cues(self) -> bool:
        return self.flag_enabled("clarify_event_transitions") or self.reports_timeline_confusion() or self.has_any(
            "장면 전환",
            "전환",
            "복도",
            "발표장",
            "흐름",
            "공간 동선",
            "인물 위치",
        )

    def prefers_sentence_simplification(self) -> bool:
        return self.prefers_stronger_scene_compaction() or self.needs_contextual_summaries() or self.has_any(
            "문장이 너무 길",
            "길고 복잡",
            "이해하기 어려",
            "이해하기 어렵",
            "가독성",
            "호흡",
        )

    def prefers_dialogue_compaction(self) -> bool:
        return self.prefers_faster_progression() or self.prefers_compact_beats() or self.has_any(
            "긴 회의",
            "회의·대화",
            "대화 장면",
            "속도감이 떨어",
            "템포가 느려",
        )

    def prefers_observable_emotion_evidence(self) -> bool:
        return self.has_any(
            "심리",
            "내면",
            "설명적",
            "감정선",
            "표정",
            "행동",
            "보여",
        )

    def wants_emotional_wave_contrast(self) -> bool:
        return self.has_any(
            "감정의 파고",
            "감정 파고",
            "감정의 고저",
            "감정 고저",
            "긴장 완화",
            "작은 유머",
            "유머",
            "친근한 묘사",
        )

    def needs_role_cues(self) -> bool:
        return self.flag_enabled("clarify_similar_character_entries") or self.has_any(
            "인물",
            "이름",
            "직책",
            "역할",
            "구분",
            "헷갈",
        )

    def needs_opening_orientation(self) -> bool:
        return self.needs_contextual_summaries() or self.has_any(
            "초반",
            "인물 설명 없이",
            "누군지",
        )

    def prefers_technical_term_restraint(self) -> bool:
        return (
            self.prefers_compact_beats()
            or self.force_reaction_after_jargon()
            or self.jargon_term_cap(default=2) <= 2
        )

    def enforces_sentence_word_cap(self, default: int = 25) -> bool:
        return self.sentence_word_cap(default=default) < default or self.has_any(
            "25단어",
            "25 단어",
            "25word",
            "긴 문장 자동 분할",
            "문장 자동 분할기",
        )

    def prefers_single_term_gloss(self) -> bool:
        return self.prefers_technical_term_restraint() and self.has_any(
            "처음 등장",
            "첫 등장",
            "첫 언급",
            "괄호",
            "정의",
            "풀어쓰기",
            "비유",
            "약어",
        )

    def prefers_stable_term_reuse(self) -> bool:
        return self.has_any("동의어", "통일", "의미 중복", "혼선")

    def prefers_list_breakup(self) -> bool:
        return self.has_any("목록", "나열", "줄바꿈", "쪼개", "분할")

    def flags_repetitive_imagery(self) -> bool:
        return self.prefers_stronger_scene_compaction() or self.has_any(
            "제스처",
            "표정",
            "손동작",
            "관찰",
            "시선",
            "행동 묘사",
        )

    def prefers_expository_dialogue_reduction(self) -> bool:
        return self.prefers_dialogue_compaction() or self.has_any(
            "정보 전달형 대사",
            "정보 전달",
            "설명 위주",
            "감정적 임팩트",
            "임팩트",
        )

    def prefers_analytical_wording_reduction(self) -> bool:
        return self.prefers_observable_emotion_evidence() or self.has_any(
            "가능성",
            "계산",
            "추론",
            "판단",
            "반복",
            "중복",
            "늘어지",
        )

    def wants_distinct_dialogue_voices(self) -> bool:
        return self.has_any("말투", "어투", "톤", "고유한 말투")

    def prefers_threat_signal_stack_compression(self) -> bool:
        return self.flag_enabled("compress_threat_signal_stack") or self.has_any(
            "위협 신호",
            "과밀",
            "인위적",
            "모니터 경보",
            "보안요원 시선",
        )


def build_reader_profile(reader_feedback: Optional[dict]) -> ReaderProfile:
    return ReaderProfile.from_feedback(reader_feedback)
