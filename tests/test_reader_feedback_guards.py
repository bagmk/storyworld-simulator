#!/usr/bin/env python3
"""
Regression tests for reader-feedback-driven readability guards.
"""

from pathlib import Path
import re
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_chapter import (
    _apply_reader_feedback_pipeline_overrides,
    _resolve_generation_style,
    _sanitize_chapter_draft_artifacts,
    adjust_scene_target_for_feedback,
)
from src.novel_writer.director import DirectorAI
from src.novel_writer.models import ClueManager, WorldState
from src.novel_writer.prose_generator import ProseGenerator
from src.novel_writer.scene_distiller import DistilledScene, SceneDistiller


class DummyLLM:
    def chat(self, *args, **kwargs):
        raise AssertionError("LLM should not be called in deterministic guard tests")


class CaptureLLM:
    def __init__(self, response: str = "ok"):
        self.response = response
        self.calls: list[dict] = []

    def chat(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


def _feedback(*items: str) -> dict:
    return {
        "what_felt_boring_or_hard": list(items),
        "style_tips": [],
        "reader_comment": " ".join(items),
        "jargon_watch_terms": ["QPU", "latency"],
        "repetition_watch_terms": ["정적", "복도 소음"],
    }


class ReaderFeedbackGuardsTest(unittest.TestCase):
    def test_scene_distiller_rebalances_mood_fragments_to_action(self):
        distiller = SceneDistiller(
            llm=DummyLLM(),
            episode_config={},
            reader_feedback=_feedback("반복되는 표현", "비슷한 상황과 묘사"),
        )
        scene = DistilledScene(
            scene_number=1,
            title="복도 대기",
            turn_range=(1, 3),
            location="복도",
            characters_present=["수민"],
            key_dialogue=[],
            key_actions=["수민은 자료를 다시 밀어 넣었다"],
            discoveries=[],
            emotional_arc="경계 -> 결심",
            beat_references=[],
            narrative_summary="잠시 정적이 흘렀다. 복도 소음이 스쳤다. 수민은 자료를 다시 밀어 넣었다.",
            pacing="building",
            raw_turn_count=3,
        )

        summary = distiller._rebalance_narrative_summary(scene)

        self.assertIn("수민은 자료를 다시 밀어 넣었다.", summary)
        self.assertNotIn("복도 소음이 스쳤다.", summary)

    def test_director_detects_repeated_technical_exchange_without_emotional_shift(self):
        director = DirectorAI(
            episode_config={"summary": "기술 브리핑", "location": "회의실"},
            world_facts={},
            clue_manager=ClueManager(),
            llm=DummyLLM(),
            reader_feedback=_feedback("기술 용어가 자주 나오고 겹친다"),
        )
        recent = [
            {"speaker_id": "a", "action_type": "dialogue", "content": "QPU latency가 다시 튀었습니다."},
            {"speaker_id": "b", "action_type": "dialogue", "content": "QPU latency와 drift가 동시에 흔들립니다."},
            {"speaker_id": "a", "action_type": "dialogue", "content": "latency 수치와 QPU 보정 파라미터를 다시 보겠습니다."},
            {"speaker_id": "b", "action_type": "dialogue", "content": "QPU drift와 latency를 그대로 유지하면 안 됩니다."},
        ]

        signal = director._scene_progress_signal(recent)

        self.assertTrue(signal["technical_stall"])
        self.assertTrue(signal["stalled"])

    def test_prose_generator_merges_clipped_sentence_runs(self):
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
            reader_feedback=_feedback("짧게 끊기는 문장이 반복된다", "비슷한 리듬"),
        )
        merged = prose._merge_clipped_sentence_runs("그의 숨이 막혔다. 문이 열렸다. 발소리가 가까워졌다.")

        self.assertEqual(len(prose._split_korean_sentences(merged)), 1)
        self.assertTrue(any(token in merged for token in ("짧은 숨이 스친 뒤", "답이 바로 나오지 않는 사이", "말끝이 가라앉자")))
        self.assertFalse(any(token in merged for token in ("그 말이 끝나자", "시선이 옮겨가자")))
        self.assertFalse(any(token in merged for token in ("그 직후", "잠시 뒤")))

    def test_rhythm_bridge_samples_avoid_flagged_stock_fragments(self):
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
            reader_feedback=_feedback("반복되는 표현"),
        )
        bridges = [prose._rhythm_bridge_sentence(i, min_chars=5, max_chars=30) for i in range(20)]
        banned = {"그의 손이 멈췄다.", "잠시 정적이 흘렀다.", "복도 소음이 스쳤다."}

        self.assertFalse(any(bridge in banned for bridge in bridges))

    def test_scene_target_adjusts_for_clipped_rhythm_and_jargon_feedback(self):
        adjusted = adjust_scene_target_for_feedback(
            target_scenes=7,
            target_words=3800,
            reader_feedback=_feedback("짧게 끊기는 문장이 계속 이어진다", "기술 용어가 자주 나온다"),
        )

        self.assertEqual(adjusted, 6)

    def test_scene_target_adjusts_for_stalled_progression_feedback(self):
        adjusted = adjust_scene_target_for_feedback(
            target_scenes=7,
            target_words=3800,
            reader_feedback=_feedback("멈춘 이유를 분석하고 고쳐봐"),
        )

        self.assertEqual(adjusted, 6)

    def test_scene_distiller_adds_merge_budget_for_stalled_progression_feedback(self):
        distiller = SceneDistiller(
            llm=DummyLLM(),
            episode_config={},
            reader_feedback=_feedback("멈춘 이유를 분석하고 고쳐봐"),
        )

        self.assertGreaterEqual(distiller._scene_merge_budget(7), 1)

    def test_director_checks_completion_earlier_when_reader_reports_stall(self):
        director = DirectorAI(
            episode_config={"summary": "회의 장면", "location": "회의실", "max_turns": 12},
            world_facts={},
            clue_manager=ClueManager(),
            llm=DummyLLM(),
            reader_feedback=_feedback("멈춘 이유를 분석하고 고쳐봐"),
        )

        self.assertEqual(director.min_turns_before_completion, 4)
        self.assertEqual(director.completion_check_interval, 1)

    def test_prose_generator_tightens_paragraph_breathing_for_stalled_progression_feedback(self):
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
            reader_feedback=_feedback("멈춘 이유를 분석하고 고쳐봐"),
        )

        controls = prose._readability_controls()

        self.assertEqual(controls["paragraph_max"], 2)

    def test_scene_distiller_clarifies_unnamed_suit_observer_vs_named_miller(self):
        distiller = SceneDistiller(
            llm=DummyLLM(),
            episode_config={},
            reader_feedback=_feedback("다크 수트 남자", "크리스찬 밀러", "같은 인물인지 헷갈린다"),
        )
        scenes = [
            DistilledScene(
                scene_number=1,
                title="메모 발견",
                turn_range=(1, 2),
                location="복도",
                characters_present=["수민", "다크 수트 남자"],
                key_dialogue=[],
                key_actions=["다크 수트 남자가 메모장을 덮었다."],
                discoveries=["다크 수트 남자의 메모에 COHERENCE가 적혀 있었다."],
                emotional_arc="경계",
                beat_references=[],
                narrative_summary="다크 수트 남자가 복도 끝에서 메모를 적고 있었다.",
                pacing="building",
                raw_turn_count=2,
            ),
            DistilledScene(
                scene_number=2,
                title="밀러 접근",
                turn_range=(3, 4),
                location="복도",
                characters_present=["수민", "Christian Miller"],
                key_dialogue=[],
                key_actions=["다크 수트 남자가 수민 곁에 멈춰 섰다."],
                discoveries=[],
                emotional_arc="압박",
                beat_references=[],
                narrative_summary="다크 수트 남자가 수민에게 조용히 말을 걸었다.",
                pacing="building",
                raw_turn_count=2,
            ),
        ]

        refined = distiller.apply_scene_guards(scenes)

        self.assertIn("이름 모를 수트 차림의 남자", refined[0].narrative_summary)
        self.assertIn("Christian Miller", refined[1].narrative_summary)

    def test_scene_distiller_stabilizes_summary_subjects_and_fragments(self):
        distiller = SceneDistiller(
            llm=DummyLLM(),
            episode_config={},
            reader_feedback=_feedback("미완 문장", "대명사 오류", "호칭 혼선"),
        )
        scene = DistilledScene(
            scene_number=1,
            title="복도 대기",
            turn_range=(1, 2),
            location="복도",
            characters_present=["수민", "Christian Miller"],
            key_dialogue=[],
            key_actions=["수민은 자료를 가방 안으로 밀어 넣었다."],
            discoveries=[],
            emotional_arc="경계",
            beat_references=[],
            narrative_summary="그는 숨을 고르지 못했다. 복도 소음.",
            pacing="building",
            raw_turn_count=2,
        )

        distiller._apply_scene_readability_guards(scene)

        self.assertIn("수민은", scene.narrative_summary)
        self.assertNotIn("복도 소음.", scene.narrative_summary)

    def test_scene_distiller_coerces_prefixed_turn_numbers(self):
        self.assertEqual(SceneDistiller._coerce_turn_number("T11"), 11)
        self.assertEqual(SceneDistiller._coerce_turn_number("turn 14"), 14)
        self.assertEqual(SceneDistiller._coerce_turn_number("₁₇"), 17)

    def test_scene_distiller_turn_range_accepts_prefixed_turn_tokens(self):
        distiller = SceneDistiller(
            llm=DummyLLM(),
            episode_config={},
            reader_feedback=_feedback("시간축", "순서가 섞여 헷갈린다"),
        )

        start, end = distiller._coerce_scene_turn_range(
            {
                "turn_start": "T11",
                "turn_end": "T14",
                "turn_range": "T11-T14",
            },
            available_turns=list(range(1, 21)),
            scene_index=0,
            total_scenes=3,
        )

        self.assertEqual((start, end), (11, 14))

    def test_scene_distiller_skips_llm_when_compact_log_exceeds_limit(self):
        llm = CaptureLLM(response="[]")
        distiller = SceneDistiller(
            llm=llm,
            episode_config={},
            runtime_policy={"distiller_prompt_chars_max": 12000},
        )
        interactions = [
            {
                "turn": i + 1,
                "speaker_name": "수민",
                "content": "압박이 길어지는 순간이 계속 이어졌다." * 3,
                "action_type": "dialogue",
            }
            for i in range(200)
        ]

        scenes = distiller._llm_distill(interactions, [], "kim_sumin", target_scenes=8)

        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(len(scenes), 8)

    def test_prose_generator_trims_post_metaphor_explanation_pairs(self):
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
            reader_feedback={
                **_feedback("비유로 분위기를 만든 직후 의미를 다시 설명한다"),
                "style_constraints": {"avoid_metaphor_explanation": 1},
            },
        )

        trimmed = prose._trim_post_metaphor_explanations(
            "그 말은 칼날처럼 얇았다. 그 뜻은 분명했다. 수민은 숨을 고쳤다."
        )

        self.assertNotIn("그 뜻은 분명했다.", trimmed)
        self.assertIn("그 말은 칼날처럼 얇았다.", trimmed)
        self.assertIn("수민은 숨을 고쳤다.", trimmed)

    def test_prose_generator_strips_meta_marker_artifacts(self):
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
            reader_feedback={
                **_feedback("메타 표식", "작업 메모"),
                "style_constraints": {"strip_meta_markers": 1},
            },
        )

        cleaned = prose._trim_post_metaphor_explanations(
            "ep01—scene21. ep01의 온도계. 수민은 숨을 고쳤다."
        )

        self.assertNotIn("ep01", cleaned.lower())
        self.assertNotIn("scene21", cleaned.lower())
        self.assertIn("수민은 숨을 고쳤다.", cleaned)

    def test_scene_distiller_merges_repeated_hallway_confrontation_beats(self):
        distiller = SceneDistiller(
            llm=DummyLLM(),
            episode_config={},
            reader_feedback={
                **_feedback("밀러와의 복도 대면이 사실상 두 번 반복된다", "하나의 대화로 압축하라"),
                "style_constraints": {"merge_repeated_confrontation_beats": 1},
            },
        )
        left = DistilledScene(
            scene_number=1,
            title="복도 질문",
            turn_range=(11, 12),
            location="복도",
            characters_present=["수민", "Christian Miller"],
            key_dialogue=[],
            key_actions=["Christian Miller가 수민에게 다가섰다."],
            discoveries=[],
            emotional_arc="경계 -> 압박",
            beat_references=[],
            narrative_summary="Christian Miller가 복도에서 수민에게 통제권을 누가 쥐는지 물었다.",
            pacing="building",
            raw_turn_count=2,
        )
        right = DistilledScene(
            scene_number=2,
            title="복도 재질문",
            turn_range=(13, 14),
            location="복도",
            characters_present=["수민", "Christian Miller"],
            key_dialogue=[],
            key_actions=["수민이 답을 늦추며 숨을 골랐다."],
            discoveries=[],
            emotional_arc="압박 -> 결심",
            beat_references=[],
            narrative_summary="Christian Miller는 같은 자리에서 책임을 누가 질지 다시 되물었다.",
            pacing="building",
            raw_turn_count=2,
        )

        self.assertTrue(distiller._scenes_need_timeline_merge(left, right))

    def test_director_detects_repeated_hallway_confrontation_exchange(self):
        director = DirectorAI(
            episode_config={"summary": "복도 대면", "location": "복도"},
            world_facts={},
            clue_manager=ClueManager(),
            llm=DummyLLM(),
            reader_feedback=_feedback("복도 대면이 사실상 두 번 반복된다"),
        )
        recent = [
            {"speaker_id": "miller", "action_type": "dialogue", "content": "Miller가 복도에서 다가서며 통제권이 누구에게 있는지 물었다."},
            {"speaker_id": "sumin", "action_type": "dialogue", "content": "수민은 같은 복도에서 바로 답하지 않았다."},
            {"speaker_id": "miller", "action_type": "dialogue", "content": "Miller는 다시 다가서며 책임을 누가 질지 되물었다."},
            {"speaker_id": "sumin", "action_type": "dialogue", "content": "수민은 답을 미루며 같은 질문을 되받았다."},
        ]

        signal = director._scene_progress_signal(recent)

        self.assertTrue(signal["repeated_concern"])
        self.assertTrue(signal["stalled"])

    def test_pipeline_overrides_enable_new_reader_feedback_flags(self):
        tuned = _apply_reader_feedback_pipeline_overrides(
            _feedback(
                "그리고, 그러자 식의 연결 문장을 줄여라",
                "비유로 분위기를 만든 직후 의미를 다시 설명한다",
                "메모 발견, 경고음, 밀러 등장처럼 장면 전환을 또렷하게",
                "다크 수트 남자와 크리스찬 밀러가 헷갈린다",
                "외부 지원, 통제권, 책임 문제 반복을 20퍼센트 압축",
            )
        )

        constraints = tuned.get("style_constraints", {})

        self.assertEqual(constraints.get("single_axis_sentences"), 1)
        self.assertEqual(constraints.get("avoid_metaphor_explanation"), 1)
        self.assertEqual(constraints.get("clarify_event_transitions"), 1)
        self.assertEqual(constraints.get("clarify_similar_character_entries"), 1)
        self.assertEqual(constraints.get("scene_compaction_ratio_target"), 80)

    def test_pipeline_overrides_capture_review_specific_style_flags(self):
        tuned = _apply_reader_feedback_pipeline_overrides(
            _feedback(
                "같은 정보와 감정이 재진술돼 제자리에서 맴도는 인상이다",
                "그 말이 끝나자, 시선이 옮겨가자 같은 연결어 반복을 줄여라",
                "메모, 모니터 경보, 보안요원 시선 같은 위협 신호가 과밀하다",
                "모레노와 밀러의 대사는 이해관계와 말버릇이 드러나야 한다",
                "핵심 문단 몇 개는 길게 호흡하라",
            )
        )

        constraints = tuned.get("style_constraints", {})

        self.assertEqual(constraints.get("single_strong_interior_beat"), 1)
        self.assertEqual(constraints.get("compress_threat_signal_stack"), 1)
        self.assertEqual(constraints.get("dialogue_agenda_contrast"), 1)
        self.assertEqual(constraints.get("prefer_pivot_paragraph_breath"), 1)
        self.assertIn("그 말이 끝나자", constraints.get("avoid_transition_terms", []))

    def test_pipeline_overrides_avoid_flagged_stock_bridge_phrase(self):
        tuned = _apply_reader_feedback_pipeline_overrides(
            _feedback("짧은 숨이 스친 뒤 같은 반복 접속구를 줄여라", "문장 리듬이 기계적이다")
        )

        constraints = tuned.get("style_constraints", {})

        self.assertIn("짧은 숨이 스친 뒤", constraints.get("avoid_transition_terms", []))
        self.assertEqual(constraints.get("max_transition_openers_per_block"), 1)

    def test_pipeline_overrides_capture_first_person_and_draft_cleanup_flags(self):
        tuned = _apply_reader_feedback_pipeline_overrides(
            _feedback("수민 1인칭 시점으로 고정하라", "미완 문장과 대명사 오류를 먼저 정리하라")
        )

        constraints = tuned.get("style_constraints", {})

        self.assertEqual(constraints.get("force_first_person_pov"), 1)
        self.assertEqual(constraints.get("force_complete_sentences"), 1)
        self.assertEqual(constraints.get("stabilize_reference_labels"), 1)

    def test_generation_style_resolves_from_reader_feedback(self):
        style = _resolve_generation_style(
            "third_person_close",
            _feedback("시리즈 컨텍스트에 맞춰 수민 1인칭 시점으로 고정하라"),
        )

        self.assertEqual(style, "first_person")

    def test_scene_distiller_compresses_overloaded_threat_signal_stack(self):
        distiller = SceneDistiller(
            llm=DummyLLM(),
            episode_config={},
            reader_feedback={
                **_feedback("메모, 모니터 경보, 보안요원 시선 같은 위협 신호가 과밀하다"),
                "style_constraints": {"compress_threat_signal_stack": 1, "max_static_threat_signals_per_scene": 1},
            },
        )
        scene = DistilledScene(
            scene_number=1,
            title="복도 경고",
            turn_range=(1, 3),
            location="복도",
            characters_present=["수민", "보안요원"],
            key_dialogue=[],
            key_actions=["수민은 문 쪽으로 반걸음 물러섰다."],
            discoveries=["메모가 의자 위에 놓여 있었다.", "모니터 경보가 복도 끝에서 울렸다."],
            emotional_arc="경계 -> 압박",
            beat_references=[],
            narrative_summary="메모가 의자 위에 놓여 있었다. 모니터 경보가 복도 끝에서 울렸다. 보안요원의 시선이 수민을 따라왔다. 수민은 문 쪽으로 반걸음 물러섰다.",
            pacing="building",
            raw_turn_count=3,
        )

        distiller._apply_scene_readability_guards(scene)

        self.assertIn("수민은 문 쪽으로 반걸음 물러섰다.", scene.narrative_summary)
        self.assertLessEqual(
            sum(
                1
                for sent in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", scene.narrative_summary)
                if distiller._threat_signal_signature(sent) and not distiller._summary_has_action_or_decision(sent)
            ),
            1,
        )

    def test_director_detects_overloaded_threat_signal_stack(self):
        director = DirectorAI(
            episode_config={"summary": "복도 압박", "location": "복도"},
            world_facts={},
            clue_manager=ClueManager(),
            llm=DummyLLM(),
            reader_feedback={
                **_feedback("메모, 모니터 경보, 보안요원 시선 같은 위협 신호가 과밀하다"),
                "style_constraints": {"compress_threat_signal_stack": 1, "max_static_threat_signals_per_scene": 1},
            },
        )
        recent = [
            {"speaker_id": "a", "action_type": "dialogue", "content": "메모가 의자 위에 놓여 있습니다."},
            {"speaker_id": "b", "action_type": "dialogue", "content": "모니터 경보가 다시 울렸습니다."},
            {"speaker_id": "a", "action_type": "dialogue", "content": "보안요원의 시선이 계속 따라옵니다."},
            {"speaker_id": "b", "action_type": "dialogue", "content": "복도 끝 화면에도 warning 표시가 떠 있습니다."},
        ]

        signal = director._scene_progress_signal(recent)

        self.assertTrue(signal["signal_stack"])
        self.assertTrue(signal["stalled"])

    def test_director_document_artifact_prompt_demands_spatial_clarity(self):
        llm = CaptureLLM("메모가 의자 위에서 눈에 들어왔다.")
        director = DirectorAI(
            episode_config={"summary": "복도 장면", "location": "복도"},
            world_facts={},
            clue_manager=ClueManager(),
            llm=llm,
            reader_feedback=_feedback("메모 발견", "공간 동선", "인물 위치"),
        )
        world = WorldState(
            current_scene="수민은 복도 벽 옆에 서 있었다.",
            location="복도",
            visible_context={"last_event": "휴식 시간이 막 시작됐다."},
        )

        director._generate_injection_event(
            "메모장에 COHERENCE와 DRIFT가 적혀 있다",
            "protagonist notices a document",
            "document_artifact",
            world,
        )

        prompt = llm.calls[-1]["kwargs"]["messages"][0]["content"]

        self.assertIn("where the protagonist is", prompt)
        self.assertIn("artifact appears", prompt)
        self.assertIn("memo", prompt.lower())

    def test_chapter_cleanup_rewrites_draft_artifacts(self):
        cleaned = _sanitize_chapter_draft_artifacts(
            "real-time viable if externally supported. 수민는 그 단어은 다시 읽었다.",
            _feedback("영어 혼입", "오탈자", "퇴고 전 원고처럼 보인다"),
        )

        self.assertNotIn("real-time viable if externally supported", cleaned)
        self.assertNotIn("수민는", cleaned)
        self.assertNotIn("단어은", cleaned)
        self.assertIn("실시간", cleaned)
        self.assertIn("수민은", cleaned)
        self.assertIn("단어는", cleaned)

    def test_prose_generator_first_person_guard_rewrites_leading_self_reference(self):
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
            reader_feedback={
                **_feedback("수민 1인칭 시점으로 고정하라"),
                "style_constraints": {"force_first_person_pov": 1},
            },
        )

        cleaned = prose._cleanup_pov_reference_artifacts(
            "수민은 숨을 골랐다. 그녀는 문 쪽을 먼저 봤다.",
            "third_person_close",
            "Kim Sumin",
        )

        self.assertIn("나는 숨을 골랐다.", cleaned)
        self.assertIn("나는 문 쪽을 먼저 봤다.", cleaned)


if __name__ == "__main__":
    unittest.main()
