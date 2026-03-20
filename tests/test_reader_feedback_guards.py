#!/usr/bin/env python3
"""
Regression tests for reader-feedback-driven readability guards.
"""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_chapter import _apply_reader_feedback_pipeline_overrides, adjust_scene_target_for_feedback
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
        self.assertTrue(any(token in merged for token in ("그 말이 끝나자", "시선이 옮겨가자", "고개를 들자", "의자가 밀리자")))
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


if __name__ == "__main__":
    unittest.main()
