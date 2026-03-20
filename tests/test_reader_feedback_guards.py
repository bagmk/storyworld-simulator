#!/usr/bin/env python3
"""
Regression tests for reader-feedback-driven readability guards.
"""

from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generate_chapter as chapter_entry
from src.novel_writer import director as director_module
from src.novel_writer import orchestrator as orchestrator_module
from src.novel_writer import scene_distiller as scene_distiller_module
from src.novel_writer import prose_generator as prose_generator_module
from generate_chapter import (
    _apply_reader_feedback_pipeline_overrides,
    _resolve_generation_style,
    _sanitize_chapter_draft_artifacts,
    adjust_scene_target_for_feedback,
)
from src.novel_writer.director import DirectorAI
from src.novel_writer.models import Agent, ClueManager, WorldState
from src.novel_writer.orchestrator import SimulationOrchestrator
from src.novel_writer.polisher import ChapterPolisher
from src.novel_writer.prose_generator import ProseGenerator
from src.novel_writer.reader_profile import build_reader_profile
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
    def test_reader_profile_centralizes_normalization_schema(self):
        profile = build_reader_profile(
            _feedback(
                "멈춘 이유를 분석하고 고쳐봐",
                "수민 1인칭 시점으로 고정하라",
                "미완 문장과 대명사 오류를 먼저 정리하라",
                "짧은 숨이 스친 뒤 같은 반복 접속구를 줄여라",
            )
        )

        self.assertTrue(profile.semantic_flags.stalled_progression)
        self.assertTrue(profile.semantic_flags.needs_draft_cleanup)
        self.assertTrue(profile.semantic_flags.prefers_sumin_first_person)
        self.assertTrue(profile.stage_hints.prose_force_first_person)
        self.assertTrue(profile.stage_hints.prose_needs_draft_cleanup)
        self.assertTrue(profile.stage_hints.chapter_reduce_scene_count)
        self.assertGreaterEqual(profile.caps.sentence_variety_window, 5)
        self.assertIn("짧은 숨이 스친 뒤", profile.term_preferences.transition_avoid_terms)
        self.assertIn("stalled_progression", profile.pattern_registry)
        self.assertTrue(profile.normalization_metadata["applied_pipeline_overrides"])

    def test_generate_chapter_feedback_helpers_delegate_to_profile_builder(self):
        class FakeProfile:
            corpus = "centralized corpus"

            def __init__(self):
                self.calls: list[tuple] = []

            def has_any(self, *tokens):
                self.calls.append(("has_any", tokens))
                return True

            def reports_stalled_progression(self):
                self.calls.append(("reports_stalled_progression",))
                return True

            def needs_draft_cleanup(self):
                self.calls.append(("needs_draft_cleanup",))
                return True

            def prefers_sumin_first_person(self):
                self.calls.append(("prefers_sumin_first_person",))
                return True

            def sanitize_chapter_draft_artifacts(self, text):
                self.calls.append(("sanitize", text))
                return "CLEANED"

            def adjusted_scene_target(self, target_scenes, target_words, *, max_words_per_scene):
                self.calls.append(("adjusted_scene_target", target_scenes, target_words, max_words_per_scene))
                return 5

            def resolve_generation_style(self, cli_style):
                self.calls.append(("resolve_generation_style", cli_style))
                return "first_person"

            def as_dict(self):
                self.calls.append(("as_dict",))
                return {"style_constraints": {"force_first_person_pov": 1}}

        fake = FakeProfile()

        with patch.object(chapter_entry, "build_reader_profile", return_value=fake) as builder:
            self.assertEqual(chapter_entry._reader_feedback_corpus({}), "centralized corpus")
            self.assertTrue(chapter_entry._reader_feedback_has_any({}, "foo"))
            self.assertTrue(chapter_entry._reader_feedback_mentions_stalled_progression({}))
            self.assertTrue(chapter_entry._reader_feedback_needs_draft_cleanup({}))
            self.assertTrue(chapter_entry._reader_feedback_prefers_sumin_first_person({}))
            self.assertEqual(chapter_entry._sanitize_chapter_draft_artifacts("raw", {}), "CLEANED")
            self.assertEqual(chapter_entry.adjust_scene_target_for_feedback(7, 3800, {}), 5)
            self.assertEqual(chapter_entry._resolve_generation_style("third_person_close", {}), "first_person")
            self.assertEqual(
                chapter_entry._apply_reader_feedback_pipeline_overrides({}),
                {"style_constraints": {"force_first_person_pov": 1}},
            )

        self.assertGreaterEqual(builder.call_count, 8)
        self.assertIn(("reports_stalled_progression",), fake.calls)
        self.assertIn(("needs_draft_cleanup",), fake.calls)
        self.assertIn(("prefers_sumin_first_person",), fake.calls)
        self.assertIn(("sanitize", "raw"), fake.calls)
        self.assertIn(("resolve_generation_style", "third_person_close"), fake.calls)

    def test_scene_distiller_llm_distill_runs_staged_contract(self):
        distiller = SceneDistiller(
            llm=DummyLLM(),
            episode_config={"summary": "복도 압박", "location": "복도"},
        )
        interactions = [
            {
                "turn": 1,
                "speaker_id": "kim_sumin",
                "speaker_name": "수민",
                "content": "\"지금 바로 확인해야 해요.\"",
                "action_type": "dialogue",
            }
        ]
        mapped = [
            DistilledScene(
                scene_number=1,
                title="복도 압박",
                turn_range=(1, 1),
                location="복도",
                characters_present=["수민"],
                key_dialogue=[],
                key_actions=["수민은 복도 끝을 확인했다."],
                discoveries=[],
                emotional_arc="경계",
                beat_references=[],
                narrative_summary="수민은 복도 끝을 확인했다.",
                pacing="building",
                raw_turn_count=1,
            )
        ]
        finalized = list(mapped)
        call_order: list[str] = []

        original_prepare = distiller._prepare_distillation_request

        with patch.object(
            distiller,
            "_prepare_distillation_request",
            side_effect=lambda *args, **kwargs: call_order.append("prepare") or original_prepare(*args, **kwargs),
        ) as prepare, patch.object(
            distiller,
            "_build_distill_prompt",
            side_effect=lambda request: call_order.append("build") or "PROMPT",
        ) as build_prompt, patch.object(
            distiller,
            "_call_distill_model",
            side_effect=lambda prompt: call_order.append("call") or "RAW",
        ) as call_model, patch.object(
            distiller,
            "_parse_distill_response",
            side_effect=lambda raw: call_order.append("parse") or [{"title": "복도 압박"}],
        ) as parse_result, patch.object(
            distiller,
            "_map_distilled_scenes",
            side_effect=lambda rows, request: call_order.append("map") or mapped,
        ) as map_scenes, patch.object(
            distiller,
            "_finalize_distilled_scenes",
            side_effect=lambda scenes, request: call_order.append("finalize") or finalized,
        ) as finalize:
            result = distiller._llm_distill(interactions, [], "kim_sumin", target_scenes=1)

        self.assertEqual(call_order, ["prepare", "build", "call", "parse", "map", "finalize"])
        self.assertIs(result, finalized)
        prepare.assert_called_once_with(interactions, [], "kim_sumin", 1)
        build_prompt.assert_called_once()
        call_model.assert_called_once_with("PROMPT")
        parse_result.assert_called_once_with("RAW")
        map_scenes.assert_called_once()
        finalize.assert_called_once()

    def test_scene_distiller_feedback_helpers_delegate_to_reader_profile(self):
        fake_profile = SimpleNamespace(
            corpus="centralized distiller corpus",
            semantic_flags=SimpleNamespace(
                prefers_compact_beats=True,
                stalled_progression=True,
                prefers_faster_progression=False,
                wants_repeated_confrontation_merge=True,
                reports_timeline_confusion=True,
                needs_contextual_summaries=False,
                prefers_stronger_scene_compaction=True,
                flags_recycled_negotiation_points=True,
                flags_stock_bridge_phrases=True,
            ),
            as_dict=lambda: {"style_constraints": {"merge_repeated_confrontation_beats": 1}},
            flag_enabled=lambda key, default=False: key == "compress_threat_signal_stack",
            static_threat_signal_cap=lambda default=2: 1,
            scene_compaction_target=lambda default=100: 75,
            summary_sentence_word_cap=lambda default=18: 14,
            force_reaction_after_jargon=lambda: True,
            summary_plain_buffer_enabled=lambda: False,
            summary_easy_metaphor_enabled=lambda: False,
            prefers_explicit_transition_cues=lambda: True,
            prefers_sentence_simplification=lambda: True,
            prefers_dialogue_compaction=lambda: False,
            prefers_observable_emotion_evidence=lambda: True,
            wants_emotional_wave_contrast=lambda: True,
            needs_role_cues=lambda: True,
            needs_opening_orientation=lambda: False,
        )

        with patch.object(scene_distiller_module, "build_reader_profile", return_value=fake_profile) as builder:
            distiller = SceneDistiller(
                llm=DummyLLM(),
                episode_config={},
                reader_feedback=_feedback("멈춘 이유를 분석하고 고쳐봐"),
            )

        self.assertEqual(distiller._reader_feedback_corpus(), "centralized distiller corpus")
        self.assertTrue(distiller._reader_prefers_compact_beats())
        self.assertTrue(distiller._reader_reports_stalled_progression())
        self.assertTrue(distiller._reader_wants_repeated_confrontation_merge())
        self.assertTrue(distiller._reader_reports_timeline_confusion())
        self.assertTrue(distiller._reader_prefers_stronger_scene_compaction())
        self.assertTrue(distiller._reader_flags_recycled_negotiation_points())
        self.assertTrue(distiller._reader_flags_stock_bridge_phrases())
        self.assertTrue(distiller._reader_prefers_explicit_transition_cues())
        self.assertTrue(distiller._reader_prefers_sentence_simplification())
        self.assertTrue(distiller._reader_prefers_observable_emotion_evidence())
        self.assertTrue(distiller._reader_wants_emotional_wave_contrast())
        self.assertTrue(distiller._reader_needs_role_cues())
        self.assertTrue(distiller._feedback_flag_enabled("compress_threat_signal_stack"))
        self.assertEqual(distiller._feedback_static_threat_signal_cap(), 1)
        self.assertEqual(distiller._feedback_scene_compaction_target(), 75)
        self.assertEqual(distiller._summary_sentence_word_cap(), 14)
        self.assertTrue(distiller._force_reaction_after_jargon())
        self.assertFalse(distiller._summary_plain_buffer_enabled())
        self.assertFalse(distiller._summary_easy_metaphor_enabled())
        self.assertEqual(
            distiller.reader_feedback,
            {"style_constraints": {"merge_repeated_confrontation_beats": 1}},
        )
        builder.assert_called_once()

    def test_scene_distiller_compresses_core_concern_lines_through_signature_pass(self):
        distiller = SceneDistiller(
            llm=DummyLLM(),
            episode_config={},
            reader_feedback=_feedback("같은 정보와 감정이 재진술돼 제자리에서 맴돈다"),
        )

        lines = [
            "밀러는 외부 지원과 통제권을 다시 물었다.",
            "밀러는 누가 통제권을 쥔 채 외부 지원을 붙일지 되물었다.",
            "수민은 명함 가장자리를 다시 눌렀다.",
        ]

        compressed = distiller._compress_core_concern_lines(lines, limit=3)

        self.assertEqual(len(compressed), 2)
        self.assertIn("수민은 명함 가장자리를 다시 눌렀다.", compressed)
        self.assertTrue(any("외부 지원" in line for line in compressed))

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

    def test_director_feedback_helpers_delegate_to_reader_profile(self):
        fake_profile = SimpleNamespace(
            as_dict=lambda: {"style_constraints": {"compress_threat_signal_stack": 1}},
            mentions=lambda *keywords: keywords == ("foo",),
            reports_stalled_progression=lambda: True,
            wants_repeated_confrontation_merge=lambda: True,
            style_constraints=lambda: {"compress_threat_signal_stack": 1},
            flag_enabled=lambda key, default=False: key == "compress_threat_signal_stack",
            tension_phrase_cap=lambda default=2: 1,
            static_threat_signal_cap=lambda default=2: 1,
        )

        with patch.object(director_module, "build_reader_profile", return_value=fake_profile) as builder:
            director = DirectorAI(
                episode_config={"summary": "복도 압박", "location": "복도"},
                world_facts={},
                clue_manager=ClueManager(),
                llm=DummyLLM(),
                reader_feedback=_feedback("foo"),
            )

        self.assertTrue(director._feedback_mentions("foo"))
        self.assertTrue(director._reader_reports_stalled_progression())
        self.assertTrue(director._reader_wants_repeated_confrontation_merge())
        self.assertEqual(director._feedback_style_constraints(), {"compress_threat_signal_stack": 1})
        self.assertTrue(director._feedback_flag_enabled("compress_threat_signal_stack"))
        self.assertEqual(director._feedback_tension_phrase_cap(), 1)
        self.assertEqual(director._feedback_static_threat_signal_cap(), 1)
        self.assertEqual(director.reader_feedback, {"style_constraints": {"compress_threat_signal_stack": 1}})
        builder.assert_called_once()

    def test_director_turn_allocator_uses_reader_profile_signals(self):
        captured: dict[str, str] = {}
        fake_profile = SimpleNamespace(
            as_dict=lambda: {},
            reports_stalled_progression=lambda: True,
            wants_repeated_confrontation_merge=lambda: False,
            prefers_technical_term_restraint=lambda: True,
            prefers_sentence_simplification=lambda: True,
            prefers_observable_emotion_evidence=lambda: True,
            prefers_stronger_scene_compaction=lambda: True,
            prefers_explicit_transition_cues=lambda: True,
            prefers_threat_signal_stack_compression=lambda: False,
            mentions=lambda *keywords: False,
            style_constraints=lambda: {},
            flag_enabled=lambda key, default=False: default,
            tension_phrase_cap=lambda default=2: default,
            static_threat_signal_cap=lambda default=2: default,
        )
        agents = [
            Agent(id="kim_sumin", name="수민", role="protagonist", bio="", invariants=[], goals=[]),
            Agent(id="miller", name="Miller", role="supporting", bio="", invariants=[], goals=[]),
        ]
        world = WorldState(active_agents=["kim_sumin", "miller"], location="복도")

        with patch.object(director_module, "build_reader_profile", return_value=fake_profile):
            director = DirectorAI(
                episode_config={"summary": "복도 압박", "location": "복도"},
                world_facts={},
                clue_manager=ClueManager(),
                llm=DummyLLM(),
                reader_feedback={},
            )

        with patch.object(
            director,
            "_scene_progress_signal",
            return_value={
                "stalled": False,
                "closure_ready": False,
                "technical_stall": False,
                "flat_tension": False,
                "explanation_loop": False,
                "repeated_concern": False,
                "signal_stack": False,
            },
        ):
            def fake_llm_call(messages, **kwargs):
                captured["prompt"] = messages[0]["content"]
                return '{"speaker_id":"kim_sumin","end_scene":false,"reason":"ok"}'

            with patch.object(director, "_safe_llm_call", side_effect=fake_llm_call):
                result = director.decide_next_speaker(
                    turn=4,
                    world=world,
                    agents=agents,
                    recent_interactions=[],
                    protagonist_id="kim_sumin",
                )

        prompt = captured["prompt"]
        self.assertEqual(result["speaker_id"], "kim_sumin")
        self.assertIn("do not spend another turn unpacking terminology", prompt)
        self.assertIn("cut low-value explanation first", prompt)
        self.assertIn("prefer turns that say one point at a time", prompt)
        self.assertIn("prefer ending the scene over extending the exchange", prompt)
        self.assertIn("end the scene early instead of extending another diagnostic turn", prompt)

    def test_orchestrator_prompt_uses_reader_profile_signals(self):
        fake_profile = SimpleNamespace(
            as_dict=lambda: {},
            repeat_terms=lambda max_terms=8: ["복도 소음"],
            jargon_terms=lambda max_terms=8: ["QPU"],
            style_constraints=lambda: {},
            term_repeat_cap=lambda default=2: 1,
            sentence_word_cap=lambda default=25: 18,
            paragraph_sentence_cap=lambda: 2,
            jargon_term_cap=lambda default=2: 1,
            mentions=lambda *keywords: False,
            prefers_sentence_simplification=lambda: True,
            enforces_sentence_word_cap=lambda default=25: True,
            prefers_technical_term_restraint=lambda: True,
            prefers_single_term_gloss=lambda: True,
            prefers_stable_term_reuse=lambda: True,
            prefers_list_breakup=lambda: True,
            flags_repetitive_imagery=lambda: True,
            needs_role_cues=lambda: True,
            prefers_observable_emotion_evidence=lambda: True,
            prefers_expository_dialogue_reduction=lambda: True,
            prefers_dialogue_compaction=lambda: True,
            prefers_explicit_transition_cues=lambda: True,
            wants_emotional_wave_contrast=lambda: True,
            prefers_analytical_wording_reduction=lambda: True,
            prefers_stronger_scene_compaction=lambda: True,
            wants_distinct_dialogue_voices=lambda: True,
        )
        agents = [
            Agent(id="kim_sumin", name="수민", role="protagonist", bio="", invariants=[], goals=[]),
            Agent(id="miller", name="Miller", role="supporting", bio="", invariants=[], goals=[]),
        ]
        world = WorldState(active_agents=["kim_sumin", "miller"], location="복도")

        with patch.object(orchestrator_module, "build_reader_profile", return_value=fake_profile) as builder:
            orchestrator = SimulationOrchestrator(
                agents=agents,
                director=SimpleNamespace(),
                world=world,
                llm=DummyLLM(),
                episode_id="ep_test",
                episode_config={"summary": "복도 압박", "location": "복도"},
                reader_feedback={},
            )

        context = {
            "goals": "상대의 의도를 확인한다.",
            "relations": {"Miller": 0.0},
            "world": world.get_context_for_agent("kim_sumin"),
            "recent": [],
            "steering": None,
        }
        _system, messages = orchestrator._build_agent_prompt(agents[0], context)
        prompt = messages[0]["content"]

        self.assertEqual(orchestrator.reader_feedback, {})
        self.assertIn("Prefer 1-2 short sentences in DIALOGUE/INNER each", prompt)
        self.assertIn("If a technical term appears this turn, mention it once and move to action/reaction.", prompt)
        self.assertIn("For the same concept, keep one stable term only", prompt)
        self.assertIn("Speaker clarity priority", prompt)
        self.assertIn("Emotion-wave priority", prompt)
        self.assertIn("Avoid repeating the same analytic words", prompt)
        self.assertIn("Reader-flagged repetition words this turn: 복도 소음.", prompt)
        builder.assert_called_once()

    def test_orchestrator_uses_reader_turn_guidance_without_legacy_helper_path(self):
        agents = [
            Agent(id="kim_sumin", name="수민", role="protagonist", bio="", invariants=[], goals=[]),
            Agent(id="miller", name="Miller", role="supporting", bio="", invariants=[], goals=[]),
        ]
        world = WorldState(active_agents=["kim_sumin", "miller"], location="복도")
        orchestrator = SimulationOrchestrator(
            agents=agents,
            director=SimpleNamespace(),
            world=world,
            llm=DummyLLM(),
            episode_id="ep_test",
            episode_config={"summary": "복도 압박", "location": "복도"},
            reader_feedback=_feedback("반복되는 표현"),
        )
        guidance = orchestrator_module.ReaderTurnGuidance(
            repeat_terms=("복도 소음",),
            jargon_terms=("QPU",),
            term_repeat_cap=1,
            sentence_word_cap=18,
            paragraph_sentence_cap=2,
            jargon_term_cap=1,
            total_cap=58,
            action_cap=18,
            dialogue_cap=28,
            inner_cap=12,
            prefers_sentence_simplification=True,
            enforces_sentence_word_cap=True,
            prefers_technical_restraint=True,
            prefers_single_term_gloss=True,
            prefers_stable_term_reuse=True,
            prefers_list_breakup=False,
            flags_repetitive_imagery=False,
            needs_role_cues=True,
            prefers_observable_emotion=True,
            prefers_expository_dialogue_reduction=False,
            prefers_dialogue_compaction=False,
            prefers_explicit_transition_cues=False,
            wants_emotional_wave_contrast=False,
            prefers_analytical_wording_reduction=False,
            wants_distinct_dialogue_voices=False,
            prefers_stronger_scene_compaction=True,
        )
        context = {
            "goals": "상대의 의도를 확인한다.",
            "relations": {"Miller": 0.0},
            "world": world.get_context_for_agent("kim_sumin"),
            "recent": [],
            "steering": None,
        }
        raw_response = (
            "TURN_MODE: dialogue\n"
            "ACTION: 수민은 문고리를 잡은 채 시선을 옮겼다.\n"
            "DIALOGUE: \"Miller, 복도 소음이 아직 남았고 복도 소음 때문에 지금은 QPU만 확인하면 됩니다.\"\n"
            "INNER: 짧게 끝내야 한다.\n"
            "EMOTION: {}\n"
            "RELATIONSHIPS: {}\n"
            "CLUES: (none)\n"
            "EXIT_SCENE: no\n"
            "AGENDA: 다음 반응을 본다.\n"
        )

        with patch.object(orchestrator, "_reader_turn_guidance", return_value=guidance), patch.object(
            orchestrator,
            "_feedback_repeat_terms",
            side_effect=AssertionError("legacy repeat helper should not be used"),
        ), patch.object(
            orchestrator,
            "_feedback_jargon_terms",
            side_effect=AssertionError("legacy jargon helper should not be used"),
        ), patch.object(
            orchestrator,
            "_reader_turn_word_caps",
            side_effect=AssertionError("legacy cap helper should not be used"),
        ), patch.object(
            orchestrator,
            "_feedback_term_repeat_cap",
            side_effect=AssertionError("legacy repeat-cap helper should not be used"),
        ), patch.object(
            orchestrator,
            "_feedback_sentence_word_cap",
            side_effect=AssertionError("legacy sentence-cap helper should not be used"),
        ):
            _system, messages = orchestrator._build_agent_prompt(agents[0], context)
            correction = orchestrator._reader_guardrail_correction(agents[0], raw_response)

        prompt = messages[0]["content"]
        self.assertIn("Reader-flagged repetition words this turn: 복도 소음.", prompt)
        self.assertIn("Limit distinct technical terms to about 1 per turn.", prompt)
        self.assertEqual(
            correction,
            "Your previous response reuses reader-flagged repeated terms too heavily. "
            "Keep only one such term and vary wording via concrete action.",
        )

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

    def test_prose_generator_generate_chapter_delegates_polish_stage_to_chapter_polisher(self):
        scene = DistilledScene(
            scene_number=1,
            title="복도 압박",
            turn_range=(1, 2),
            location="복도",
            characters_present=["수민", "Christian Miller"],
            key_dialogue=[],
            key_actions=["수민은 문 쪽으로 반걸음 물러섰다."],
            discoveries=["Christian Miller의 명함이 보였다."],
            emotional_arc="경계 -> 압박",
            beat_references=[],
            narrative_summary="수민은 문 쪽으로 반걸음 물러섰다.",
            pacing="building",
            raw_turn_count=2,
        )
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test", "location": "복도"},
        )
        polish_calls: list[dict] = []

        def fake_polish(text, target_words, style, protagonist_name, chapter_anchors, prose_adapter):
            polish_calls.append(
                {
                    "text": text,
                    "target_words": target_words,
                    "style": style,
                    "protagonist_name": protagonist_name,
                    "chapter_anchors": list(chapter_anchors or []),
                    "adapter_is_prose": prose_adapter is prose,
                }
            )
            return "FINAL POLISHED CHAPTER"

        prose.chapter_polisher = SimpleNamespace(polish_chapter=fake_polish)

        with patch.object(prose, "_calculate_scene_budgets", return_value=[180]), patch.object(
            prose,
            "_build_episode_context",
            return_value={
                "episode_number": 1,
                "total_episodes": 49,
                "location": "복도",
                "date": "",
                "summary": "복도 압박",
                "pacing": "normal",
                "pacing_tone": "balanced",
                "protagonist": "Kim Sumin",
                "beats": [],
                "beat_by_id": {},
                "recommended_length": 800,
            },
        ), patch.object(prose, "_build_previous_episode_context", return_value=""), patch.object(
            prose,
            "_generate_title",
            return_value="복도 압박",
        ), patch.object(
            prose,
            "_generate_scene_prose",
            return_value="SCENE DRAFT",
        ), patch.object(
            prose,
            "_extract_anchor_terms",
            return_value=["ANCHOR_TOKEN"],
        ), patch.object(
            prose,
            "_collect_episode_anchor_terms",
            return_value=["ANCHOR_TOKEN"],
        ), patch.object(
            prose,
            "_select_anchor_terms_for_coverage",
            return_value=["ANCHOR_TOKEN"],
        ), patch.object(
            prose,
            "_tune_coverage_anchors",
            return_value=["ANCHOR_TOKEN"],
        ), patch.object(
            prose,
            "_combine_with_transitions",
            return_value="COMBINED DRAFT",
        ), patch.object(
            prose,
            "_collect_style_diagnostics",
            return_value={
                "avg_paragraph_sentences": 2.0,
                "long_sentence_ratio": 0.0,
                "jargon_repeat_terms": 0,
                "max_visual_streak": 1,
            },
        ), patch.object(prose, "_write_chapter") as write_chapter:
            out_path = prose.generate_chapter(
                [scene],
                protagonist_name="Kim Sumin",
                style="third_person_close",
                target_words=800,
            )

        self.assertEqual(out_path.endswith("ep_test_chapter.txt"), True)
        self.assertEqual(len(polish_calls), 1)
        self.assertEqual(polish_calls[0]["text"], "COMBINED DRAFT")
        self.assertEqual(polish_calls[0]["target_words"], 800)
        self.assertEqual(polish_calls[0]["style"], "third_person_close")
        self.assertEqual(polish_calls[0]["protagonist_name"], "Kim Sumin")
        self.assertEqual(polish_calls[0]["chapter_anchors"], ["ANCHOR_TOKEN"])
        self.assertTrue(polish_calls[0]["adapter_is_prose"])
        self.assertEqual(write_chapter.call_args[0][2], "FINAL POLISHED CHAPTER")

    def test_prose_generator_polish_wrappers_delegate_to_public_chapter_polisher_stages(self):
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
        )
        stage_calls: list[tuple[str, tuple, dict]] = []

        def record(name, result):
            def inner(*args, **kwargs):
                stage_calls.append((name, args, kwargs))
                return result
            return inner

        prose.chapter_polisher = SimpleNamespace(
            run_llm_polish=record("llm", "POLISHED"),
            ensure_anchor_coverage=record("anchors", "ANCHOR_COVERED"),
            apply_reader_feedback_pass=record("feedback", "FEEDBACK_POLISHED"),
        )

        self.assertEqual(
            prose._polish("DRAFT", 800, "third_person_close", ["ANCHOR"]),
            "POLISHED",
        )
        self.assertEqual(
            prose._ensure_anchor_coverage("DRAFT", ["ANCHOR"], 800, "third_person_close"),
            "ANCHOR_COVERED",
        )
        self.assertEqual(
            prose._reader_feedback_final_pass("DRAFT", 800, "third_person_close", ["ANCHOR"]),
            "FEEDBACK_POLISHED",
        )
        self.assertEqual([name for name, _args, _kwargs in stage_calls], ["llm", "anchors", "feedback"])
        self.assertIs(stage_calls[0][2]["prose_adapter"], prose)
        self.assertIs(stage_calls[1][2]["prose_adapter"], prose)
        self.assertIs(stage_calls[2][2]["prose_adapter"], prose)

    def test_prose_generator_feedback_helpers_delegate_to_reader_profile(self):
        fake_profile = SimpleNamespace(
            semantic_flags=SimpleNamespace(stalled_progression=True),
            mentions=lambda *keywords: keywords == ("foo",),
            repeat_terms=lambda max_terms=10: ["복도 소음"],
            jargon_terms=lambda max_terms=10: ["QPU"],
            style_constraints=lambda: {"force_first_person_pov": 1},
            flag_enabled=lambda key, default=False: key == "force_first_person_pov",
            term_repeat_cap=lambda default=2: 1,
            sentence_word_cap=lambda default=25: 18,
            paragraph_sentence_cap=lambda: 2,
            dense_sentence_cap=lambda default=2: 1,
            jargon_term_cap=lambda default=2: 1,
            sensory_channel_cap=lambda default=2: 1,
            emotion_repeat_cap=lambda default=1: 1,
            transition_char_window=lambda: (10, 15),
            short_beat_char_window=lambda: (14, 28),
            short_beats_per_scene=lambda: (0, 1),
            transition_opener_cap=lambda default=2: 1,
            transition_avoid_terms=lambda: {"그리고"},
            sentence_variety_window=lambda default=4: 5,
            needs_draft_cleanup=lambda: True,
            as_dict=lambda: {"style_constraints": {"force_first_person_pov": 1}},
        )

        with patch.object(prose_generator_module, "build_reader_profile", return_value=fake_profile) as builder:
            prose = ProseGenerator(
                llm=DummyLLM(),
                episode_config={"id": "ep_test"},
                reader_feedback=_feedback("foo"),
            )

        self.assertTrue(prose._feedback_reports_stalled_progression())
        self.assertTrue(prose._feedback_mentions("foo"))
        self.assertEqual(prose._feedback_repeat_terms(), ["복도 소음"])
        self.assertEqual(prose._feedback_jargon_terms(), ["QPU"])
        self.assertEqual(prose._feedback_style_constraints(), {"force_first_person_pov": 1})
        self.assertTrue(prose._feedback_flag_enabled("force_first_person_pov"))
        self.assertEqual(prose._feedback_term_repeat_cap(), 1)
        self.assertEqual(prose._feedback_sentence_word_cap(), 18)
        self.assertEqual(prose._feedback_paragraph_sentence_cap(), 2)
        self.assertEqual(prose._feedback_dense_sentence_cap(), 1)
        self.assertEqual(prose._feedback_jargon_term_cap(), 1)
        self.assertEqual(prose._feedback_sensory_channel_cap(), 1)
        self.assertEqual(prose._feedback_emotion_repeat_cap(), 1)
        self.assertEqual(prose._feedback_transition_char_window(), (10, 15))
        self.assertEqual(prose._feedback_short_beat_char_window(), (14, 28))
        self.assertEqual(prose._feedback_short_beats_per_scene(), (0, 1))
        self.assertEqual(prose._feedback_transition_opener_cap(), 1)
        self.assertEqual(prose._feedback_transition_avoid_terms(), {"그리고"})
        self.assertEqual(prose._feedback_sentence_variety_window(), 5)
        self.assertTrue(prose._feedback_needs_draft_cleanup())
        self.assertEqual(prose.reader_feedback, {"style_constraints": {"force_first_person_pov": 1}})
        builder.assert_called_once()

    def test_chapter_polisher_runs_explicit_stage_pipeline_in_order(self):
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
        )
        polisher = ChapterPolisher(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
        )
        call_order: list[str] = []

        with patch.object(
            polisher,
            "run_llm_polish",
            side_effect=lambda *args, **kwargs: call_order.append("llm") or "LLM_POLISHED",
        ) as llm_stage, patch.object(
            polisher,
            "ensure_anchor_coverage",
            side_effect=lambda *args, **kwargs: call_order.append("anchors") or "ANCHOR_FIXED",
        ) as anchor_stage, patch.object(
            polisher,
            "apply_reader_feedback_pass",
            side_effect=lambda *args, **kwargs: call_order.append("feedback") or "FEEDBACK_FIXED",
        ) as feedback_stage, patch.object(
            polisher,
            "apply_deterministic_cleanup",
            side_effect=lambda *args, **kwargs: call_order.append("cleanup") or "FINAL_TEXT",
        ) as cleanup_stage:
            result = polisher.polish_chapter(
                "DRAFT",
                target_words=800,
                style="third_person_close",
                protagonist_name="Kim Sumin",
                chapter_anchors=["ANCHOR"],
                prose_adapter=prose,
            )

        self.assertEqual(result, "FINAL_TEXT")
        self.assertEqual(call_order, ["llm", "anchors", "feedback", "cleanup"])
        llm_stage.assert_called_once_with("DRAFT", 800, "third_person_close", ["ANCHOR"], prose)
        anchor_stage.assert_called_once_with("LLM_POLISHED", ["ANCHOR"], 800, "third_person_close", prose)
        feedback_stage.assert_called_once_with("ANCHOR_FIXED", 800, "third_person_close", ["ANCHOR"], prose)
        cleanup_stage.assert_called_once_with("FEEDBACK_FIXED", "third_person_close", "Kim Sumin", prose)

    def test_chapter_polisher_deterministic_cleanup_preserves_beat_order(self):
        prose = ProseGenerator(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
        )
        polisher = ChapterPolisher(
            llm=DummyLLM(),
            episode_config={"id": "ep_test"},
        )
        text = "ALPHA_BEAT 수민은 숨을 골랐다.\n\nBETA_BEAT 밀러가 명함을 내밀었다."

        cleaned = polisher.apply_deterministic_cleanup(
            text,
            style="third_person_close",
            protagonist_name="Kim Sumin",
            prose_adapter=prose,
        )

        self.assertIn("ALPHA_BEAT", cleaned)
        self.assertIn("BETA_BEAT", cleaned)
        self.assertLess(cleaned.index("ALPHA_BEAT"), cleaned.index("BETA_BEAT"))
        self.assertNotIn("GAMMA_BEAT", cleaned)

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
